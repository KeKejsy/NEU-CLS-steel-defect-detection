"""成员 C · 检测网络：CSPResNet 主干（PP-YOLOE-s 用）

与 Darknet53 一样，这里**逐层展开写死**，不做循环 + 通道表推导。
原因参见 darknet.py 的说明：循环推导时最容易把「下采样卷积的输出通道」与
「CSP 层的输入通道」搞混，导致 conv 权重形状与输入对不上而报错。

结构（640x640 输入，下采样共 5 次：stem 1 次 + 阶段间 4 次）：

    层        类型                      通道         分辨率   步长
    stem      Conv 3x3 /s2              3->64        320      /2
    stage1    Conv 3x3 /s2   64->128    128          160      /4
              CSPLayer(128, 3 块) + ESE
    stage2    Conv 3x3 /s2  128->256    256           80      /8    -> f3
              CSPLayer(256, 3 块) + ESE
    stage3    Conv 3x3 /s2  256->512    512           40      /16   -> f4
              CSPLayer(512, 6 块) + ESE
    stage4    Conv 3x3 /s2  512->1024  1024           20      /32   -> f5
              CSPLayer(1024, 3 块) + ESE

**与 Darknet53 一样的坑**：YOLO 系检测只要 stride 8/16/32 三个尺度，
所以从输入到 f5 必须恰好是 5 次下采样。此前我把 stem 写成两次下采样（到 /4）
却只留了 3 个 stage，整体错位一级，f3 变成了 /16、尺寸从 80 变成 40。
现在统一成「stem 一次 + 每个 stage 一次」，共 5 次。

关键约束：CSPLayer 的 out_ch 必须等于「紧邻它的下采样卷积的输出通道」，
也就是送入 CSPLayer 的特征通道（CSPLayer 本身不改变通道数）。
"""

import paddle.nn as nn

from core.layers import ConvBNLayer, CSPLayer, ESEAttention


class CSPStage(nn.Layer):
    """一个 CSP 阶段：3x3/s2 下采样 + CSPLayer + ESE 注意力

    下采样卷积把通道从 in_ch 变到 down_ch，CSPLayer 再在 down_ch 上做残差堆叠
    并输出 out_ch（本实现里 down_ch == out_ch，保持通道不变以便残差相加）。
    """

    def __init__(self, in_ch, down_ch, out_ch, num_blocks, act="silu", use_ese=True):
        super().__init__()
        self.down = ConvBNLayer(in_ch, down_ch, 3, stride=2, act=act)
        self.csp = CSPLayer(down_ch, out_ch, num_blocks=num_blocks,
                            shortcut=True, act=act, use_ese=False)
        self.ese = ESEAttention(out_ch) if use_ese else None

    def forward(self, x):
        x = self.down(x)
        x = self.csp(x)
        if self.ese is not None:
            x = self.ese(x)
        return x


class CSPResNet(nn.Layer):
    """CSPResNet 主干，返回 stride 8 / 16 / 32 三个尺度特征（256 / 512 / 1024 通道）"""

    def __init__(self, base_channels=64, depths=(3, 3, 6, 3), act="silu",
                 use_ese=True, return_idx=(1, 2, 3)):
        super().__init__()
        b = base_channels  # 64
        self.return_idx = list(return_idx)

        # stem：一次下采样到 /2
        self.stem = ConvBNLayer(3, b, 3, stride=2, act=act)         # 64   /2

        # 四个阶段：/4、/8、/16、/32，通道 128 / 256 / 512 / 1024
        self.stage1 = CSPStage(b, b * 2, b * 2, depths[0], act, use_ese)        # 128 /4
        self.stage2 = CSPStage(b * 2, b * 4, b * 4, depths[1], act, use_ese)    # 256 /8
        self.stage3 = CSPStage(b * 4, b * 8, b * 8, depths[2], act, use_ese)    # 512 /16
        self.stage4 = CSPStage(b * 8, b * 16, b * 16, depths[3], act, use_ese)  # 1024 /32

        self.out_channels = [b * 4, b * 8, b * 16]  # 256 / 512 / 1024

    def forward(self, x):
        x = self.stem(x)
        feats = []
        for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
            x = stage(x)
            feats.append(x)
        return [feats[i] for i in self.return_idx]
