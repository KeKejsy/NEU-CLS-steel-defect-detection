"""成员 C · 检测网络

两个待对比的网络：
    PP-YOLOE-s   anchor-free，DFL 回归，ESE 注意力（本项目中作为更强的一方）
    YOLOv3       anchor-based，三尺度 FPN，GIoU 回归（经典基线）

两者共用 core/ 下的数据管线、坐标约定与后处理，差异只体现在网络结构与
回归方式上，这样对比结果才能归因到「网络设计」而不是「数据口径不同」。
"""

from .ppyoloe import PPYOLOES
from .yolov3 import YOLOv3

NETWORKS = {
    "ppyoloe_s": PPYOLOES,
    "yolov3": YOLOv3,
}

__all__ = ["PPYOLOES", "YOLOv3", "NETWORKS"]
