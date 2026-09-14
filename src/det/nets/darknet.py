"""成员 C · 检测网络：Darknet53 主干（YOLOv3 用）

结构说明（这张表是核对依据，改结构必须同步改表）：

    层        类型                  通道          224 输入   步长
    conv0     Conv 3x3 /s2          3->32          112       /2
    conv1     Conv 3x3 /s2         32->64           56       /4
    conv2     Conv 3x3 /s2         64->128          28       /8    -> f3 (256 通道级)
    stage2    1x1->64, Res x2, 1x1->128            28       /8
    conv3     Conv 3x3 /s2        128->256          14       /16
    stage3    1x1->128, Res x8, 1x1->256           14       /16
    conv4     Conv 3x3 /s2        256->512           7       /32
    stage4    1x1->256, Res x8, 1x1->512            7       /32
    conv5     Conv 3x3 /s2        512->1024          4       /64   （不再返回）
    stage5    1x1->512, Res x4, 1x1->1024           4       /64

**下采样次数是本文件最容易写错的地方**：YOLOv3 只取 stride 8/16/32 三个尺度，
所以从输入到 f5 恰好是 5 次下采样（conv0..conv4）。
我此前多写了一层（stage1 占了一个 /4 却又没有对应的下采样卷积），
导致 f3/f4/f5 整体错位到 /16、/32、/64，尺寸从 28/14/7 变成 14/7/4。
因此本文件**把 YOLOv3 真正用到的 5 个 3x3/s2 卷积只写 5 个**，
stage1 这类不改变分辨率的层不再单独占一次下采样。

返回的三个特征图取自 conv2 / conv3 / conv4 的输出（即各 stage 之前的那个下采样卷积），
通道数依次 256 / 512 / 1024，与 YOLOv3 官方 neck 的输入要求一致。
"""

import paddle.nn as nn

from core.layers import Bottleneck, ConvBNLayer


class ResidualStage(nn.Layer):
    """一个残差阶段：1x1 降到 hidden -> num_blocks 个瓶颈块 -> 1x1 升回 out_ch

    通道变化封闭在本阶段内，分辨率不变，这样「通道」和「分辨率」两种变化
    各归一处，不会再出现数量错位。
    """

    def __init__(self, in_ch, hidden_ch, out_ch, num_blocks, act="leaky"):
        super().__init__()
        layers = [ConvBNLayer(in_ch, hidden_ch, 1, act=act)]
        layers += [Bottleneck(hidden_ch, hidden_ch, shortcut=True, expansion=1.0, act=act)
                   for _ in range(num_blocks)]
        layers.append(ConvBNLayer(hidden_ch, out_ch, 1, act=act))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class Darknet53(nn.Layer):
    """Darknet53 主干，返回 stride 8 / 16 / 32 三个尺度特征图（256 / 512 / 1024 通道）"""

    def __init__(self, norm_act="leaky", base_channels=32, use_csp=False):
        super().__init__()
        act = norm_act
        b = base_channels  # 32

        # 5 次下采样：conv0 -> /2, conv1 -> /4, conv2 -> /8, conv3 -> /16, conv4 -> /32
        self.conv0 = ConvBNLayer(3, b, 3, stride=2, act=act)              # 32   /2
        self.conv1 = ConvBNLayer(b, b * 2, 3, stride=2, act=act)          # 64   /4
        self.conv2 = ConvBNLayer(b * 2, b * 8, 3, stride=2, act=act)      # 256  /8   -> f3
        self.conv3 = ConvBNLayer(b * 8, b * 16, 3, stride=2, act=act)     # 512  /16  -> f4
        self.conv4 = ConvBNLayer(b * 16, b * 32, 3, stride=2, act=act)    # 1024 /32  -> f5
        # conv5 只用于给 stage5 提供更深语义，不返回（保持与官方层数一致）
        self.conv5 = ConvBNLayer(b * 32, b * 32, 3, stride=2, act=act)    # 1024 /64

        # 残差阶段（官方 idx：stage2=2 块, stage3=8 块, stage4=8 块, stage5=4 块）
        self.stage2 = ResidualStage(b * 8, b * 4, b * 8, 2, act)          # 256
        self.stage3 = ResidualStage(b * 16, b * 8, b * 16, 8, act)        # 512
        self.stage4 = ResidualStage(b * 32, b * 16, b * 32, 8, act)       # 1024
        self.stage5 = ResidualStage(b * 32, b * 16, b * 32, 4, act)       # 1024

        self.out_channels = [b * 8, b * 16, b * 32]  # 256 / 512 / 1024

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        f3 = self.conv2(x)          # /8   256 通道
        x = self.stage2(f3)

        f4 = self.conv3(x)          # /16  512 通道
        x = self.stage3(f4)

        f5 = self.conv4(x)          # /32  1024 通道
        x = self.stage4(f5)

        x = self.conv5(x)           # /64
        x = self.stage5(x)

        return f3, f4, f5
