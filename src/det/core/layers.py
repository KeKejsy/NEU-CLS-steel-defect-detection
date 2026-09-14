"""成员 C · 检测核心：网络基础层

两个主干网络（Darknet53 与 CSPResNet）共用的积木都放这里：
    ConvBNLayer / ConvBNSiLU / Bottleneck / CSPLayer / SPPF / ESEAttention

设计原则：参数量、通道数全部由调用方传入，本模块不预设任何具体网络结构，
这样 YOLOv3 与 PP-YOLOE-s 可以复用同一套层，避免两份重复代码各自演化。
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


# --------------------------------------------------------------------------
# 基础卷积单元
# --------------------------------------------------------------------------
class ConvBNLayer(nn.Layer):
    """Conv2D + BN (+ 激活)，检测网络里出现频率最高的组合

    act=None 时只做卷积+BN，用于检测头最后一层（要输出 logits，不能加激活）。
    """

    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=None,
                 groups=1, act="silu", bias=False):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2D(in_ch, out_ch, kernel_size, stride, padding,
                              groups=groups, bias_attr=bias)
        self.bn = nn.BatchNorm2D(out_ch)
        self.act_name = act

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.act_name == "silu":
            return F.silu(x)
        if self.act_name == "mish":
            return x * paddle.tanh(F.softplus(x))
        if self.act_name == "leaky":
            return F.leaky_relu(x, negative_slope=0.1)
        if self.act_name == "relu":
            return F.relu(x)
        return x


class Bottleneck(nn.Layer):
    """标准瓶颈块：1x1 降维 -> 3x3 卷积 -> 残差相加

    shortcut=False 用于最外层 CSP 分支（该处通道数刚被 1x1 压过，直接相加会出错）。
    """

    def __init__(self, in_ch, out_ch, shortcut=True, groups=1, expansion=0.5, act="silu"):
        super().__init__()
        hidden = int(out_ch * expansion)
        self.conv1 = ConvBNLayer(in_ch, hidden, 1, act=act)
        self.conv2 = ConvBNLayer(hidden, out_ch, 3, groups=groups, act=act)
        self.use_shortcut = shortcut and in_ch == out_ch

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.use_shortcut else y


class SPPF(nn.Layer):
    """串行三次 5x5 最大池化，等价于 SPP 的感受野但更快

    目的：把局部特征汇聚成多尺度上下文，对 crazing（龟裂，细密纹理）这类
    大范围缺陷有帮助。
    """

    def __init__(self, in_ch, out_ch, kernel_size=5, act="silu"):
        super().__init__()
        hidden = in_ch // 2
        self.conv1 = ConvBNLayer(in_ch, hidden, 1, act=act)
        self.conv2 = ConvBNLayer(hidden * 4, out_ch, 1, act=act)
        self.pool = nn.MaxPool2D(kernel_size=kernel_size, stride=1,
                                 padding=kernel_size // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(paddle.concat([x, y1, y2, y3], axis=1))


class ESEAttention(nn.Layer):
    """Effective Squeeze-Excitation：通道注意力

    PP-YOLOE 用它替换原版 SE。'effective' 指省掉了 SE 里的降维再升维，
    只用一个与输入等宽的全连接，参数更少而效果相当。
    """

    def __init__(self, channels, act="relu"):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Conv2D(channels, channels, 1, bias_attr=True)
        self.act_name = act

    def forward(self, x):
        y = self.pool(x)
        y = self.fc(y)
        if self.act_name == "relu":
            y = F.relu(y)
        return x * F.sigmoid(y)


# --------------------------------------------------------------------------
# CSP 结构：两套主干共用，靠参数区分
# --------------------------------------------------------------------------
class CSPLayer(nn.Layer):
    """CSP 瓶颈层：输入分两路，一路过 n 个 Bottleneck，另一路直连，最后拼接

    Darknet53（YOLOv3）与 CSPResNet（PP-YOLOE）的残差阶段都是这个结构，
    差别只在 expansion / groups / 是否加 ESE，所以统一成一个类。

    use_ese=True 时在输出处加通道注意力（PP-YOLOE 的做法）。
    """

    def __init__(self, in_ch, out_ch, num_blocks=1, shortcut=True, groups=1,
                 expansion=0.5, act="silu", use_ese=False):
        super().__init__()
        hidden = out_ch // 2
        self.conv1 = ConvBNLayer(in_ch, hidden, 1, act=act)
        self.conv2 = ConvBNLayer(in_ch, hidden, 1, act=act)
        self.conv3 = ConvBNLayer(hidden * 2, out_ch, 1, act=act)
        self.blocks = nn.Sequential(*[
            Bottleneck(hidden, hidden, shortcut=shortcut, groups=groups,
                       expansion=expansion, act=act)
            for _ in range(num_blocks)
        ])
        self.ese = ESEAttention(out_ch) if use_ese else None

    def forward(self, x):
        y1 = self.blocks(self.conv1(x))
        y2 = self.conv2(x)
        y = self.conv3(paddle.concat([y1, y2], axis=1))
        if self.ese is not None:
            y = self.ese(y)
        return y


class RepConv(nn.Layer):
    """重参数化卷积（训练时 3x3 + 1x1 + 恒等三路并行）

    PP-YOLOE 在 neck 里用它：训练时多路并行增强梯度，推理时可合并成单个 3x3，
    这里只保留训练形态（本作业不做部署期融合，报告里会说明这一点）。
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None,
                 groups=1, act="silu"):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv1 = ConvBNLayer(in_ch, out_ch, kernel_size, stride, padding,
                                 groups=groups, act=None)
        self.conv2 = ConvBNLayer(in_ch, out_ch, 1, stride, 0, groups=groups, act=None)
        self.has_identity = (in_ch == out_ch and stride == 1)
        if self.has_identity:
            self.bn_identity = nn.BatchNorm2D(in_ch)
        self.act_name = act

    def forward(self, x):
        y = self.conv1(x) + self.conv2(x)
        if self.has_identity:
            y = y + self.bn_identity(x)
        if self.act_name == "silu":
            return F.silu(y)
        if self.act_name == "leaky":
            return F.leaky_relu(y, negative_slope=0.1)
        return y


class CSPRepLayer(nn.Layer):
    """PP-YOLOE neck 里的 CSP 层，瓶颈块换成 RepConv"""

    def __init__(self, in_ch, out_ch, num_blocks=3, expansion=1.0, act="silu"):
        super().__init__()
        hidden = int(out_ch * expansion)
        self.conv1 = ConvBNLayer(in_ch, hidden, 1, act=act)
        self.conv2 = ConvBNLayer(in_ch, hidden, 1, act=act)
        self.conv3 = ConvBNLayer(hidden * 2, out_ch, 1, act=act)
        self.blocks = nn.Sequential(*[
            RepConv(hidden, hidden, 3, act=act) for _ in range(num_blocks)
        ])

    def forward(self, x):
        y1 = self.blocks(self.conv1(x))
        y2 = self.conv2(x)
        return self.conv3(paddle.concat([y1, y2], axis=1))
