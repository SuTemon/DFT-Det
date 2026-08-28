# Copyright (c) FusionKD. All rights reserved.
"""
DualTeacherDistillationScheduler
=================================
管理两位教师的蒸馏权重随训练进度动态变化：

  Teacher-1 (Online Teacher)  : GLIP/GroundingDINO，在线前向，提供软标签 + 特征 + 文本对齐蒸馏
  Teacher-2 (Pseudo Teacher)  : 离线静态伪标签，提供 quality-score 加权的 hard-label 检测 loss

三阶段调度策略
--------------
Stage-1  [0, warmup_end)        强 T1 软蒸馏 + 强 T2 伪标签      → 快速冷启动
Stage-2  [warmup_end, decay_end) T1 线性衰减，T2 缓慢衰减         → 逐步自主
Stage-3  [decay_end, total)      T1 轻度软蒸馏，T2 进一步衰减      → Student 主导

性能差距自适应 (可选)
---------------------
每次 val 后调用 update_performance()，
若 Student 与 T1 差距仍大则自动提升 T1 权重，防止提前放弃蒸馏。
"""

import math
from typing import Dict, Optional


class DualTeacherDistillationScheduler:
    """三阶段双教师动态蒸馏权重调度器。

    Args:
        total_epochs (int): 总训练轮数。
        warmup_ratio (float): 阶段1结束比例，默认0.30。
        decay_ratio  (float): 阶段2结束比例，默认0.70。
        t1_min_weight (float): T1 (Online) 权重下限。
        t2_min_weight (float): T2 (Pseudo) 权重下限。
        gap_boost_max (float): 性能差距带来的最大权重补偿。
        schedule_mode (str): 'linear' | 'cosine'，阶段2的衰减曲线形状。
    """

    # ------------------------------------------------------------------ #
    #  权重含义说明                                                          #
    # ------------------------------------------------------------------ #
    # t1_soft_cls   : T1 → Student 分类软标签 KD 权重
    # t1_soft_reg   : T1 → Student 回归软标签 KD 权重
    # t1_soft_ctr   : T1 → Student 中心度 KD 权重
    # t1_feat       : T1 → Student 多尺度特征对齐权重
    # t1_text_align : T1 → Student 文本-视觉响应图对齐权重
    # t2_pseudo_det : T2 伪标签 hard-label 检测 loss 权重
    # det           : Student 自身检测 loss 权重（始终 1.0）

    def __init__(
        self,
        total_epochs: int,
        warmup_ratio: float = 0.30,
        decay_ratio: float = 0.70,
        t1_min_weight: float = 0.20,
        t2_min_weight: float = 0.10,
        gap_boost_max: float = 0.40,
        schedule_mode: str = 'cosine',
    ):
        assert 0 < warmup_ratio < decay_ratio < 1.0
        self.total_epochs = total_epochs
        self.warmup_end   = int(total_epochs * warmup_ratio)
        self.decay_end    = int(total_epochs * decay_ratio)
        self.t1_min       = t1_min_weight
        self.t2_min       = t2_min_weight
        self.gap_boost_max = gap_boost_max
        self.schedule_mode = schedule_mode

        # 运行时状态
        self.current_epoch   = 0
        self._student_map    = None   # 最近一次 val map50
        self._teacher_map    = None   # T1 在同一数据集的 map50

        # 缓存当前 weights，便于外部读取
        self._weights: Dict[str, float] = {}
        self._stage: int = 1

    # ------------------------------------------------------------------ #
    #  公开 API                                                             #
    # ------------------------------------------------------------------ #

    def step(self, epoch: int) -> Dict[str, float]:
        """在每个 epoch 开始前调用，返回本 epoch 的权重字典。"""
        self.current_epoch = epoch
        base_t1, base_t2 = self._base_weights(epoch)

        # 性能差距自适应提升
        t1_boost = self._gap_boost()
        base_t1 = min(1.0, base_t1 + t1_boost)

        self._weights = self._build_weight_dict(base_t1, base_t2)
        self._stage   = self._current_stage(epoch)
        return self._weights

    def update_performance(
        self,
        student_map50: float,
        teacher_map50: Optional[float] = None,
    ):
        """每次 val 后调用，记录性能用于自适应调整。"""
        self._student_map = student_map50
        if teacher_map50 is not None:
            self._teacher_map = teacher_map50

    @property
    def weights(self) -> Dict[str, float]:
        """返回最近一次 step() 计算的权重字典。"""
        return self._weights

    @property
    def stage(self) -> int:
        """返回当前训练阶段 (1/2/3)。"""
        return self._stage

    def log_info(self) -> str:
        """返回可直接打印的调度状态字符串。"""
        w = self._weights
        gap = (
            f"{self._teacher_map:.1f}->{self._student_map:.1f}"
            if self._teacher_map and self._student_map
            else "N/A"
        )
        return (
            f"[DualScheduler] epoch={self.current_epoch} stage={self._stage} "
            f"gap={gap} | "
            f"t1_cls={w.get('t1_soft_cls', 0):.2f} "
            f"t1_feat={w.get('t1_feat', 0):.2f} "
            f"t1_txt={w.get('t1_text_align', 0):.2f} "
            f"t2_pseudo={w.get('t2_pseudo_det', 0):.2f}"
        )

    # ------------------------------------------------------------------ #
    #  内部方法                                                             #
    # ------------------------------------------------------------------ #

    def _current_stage(self, epoch: int) -> int:
        if epoch < self.warmup_end:
            return 1
        if epoch < self.decay_end:
            return 2
        return 3

    def _base_weights(self, epoch: int):
        """
        返回 (base_t1, base_t2)，范围均在 [min, 1.0]。

        Stage-1: 均为 1.0
        Stage-2: 按 schedule_mode 从 1.0 衰减到 min
        Stage-3: 固定在 min
        """
        if epoch < self.warmup_end:
            return 1.0, 1.0

        if epoch >= self.decay_end:
            return self.t1_min, self.t2_min

        # Stage-2: 线性 or 余弦衰减
        progress = (epoch - self.warmup_end) / max(
            self.decay_end - self.warmup_end, 1
        )  # [0, 1)

        if self.schedule_mode == 'cosine':
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1→0
        else:  # 'linear'
            factor = 1.0 - progress  # 1→0

        base_t1 = self.t1_min + (1.0 - self.t1_min) * factor
        base_t2 = self.t2_min + (1.0 - self.t2_min) * factor
        return base_t1, base_t2

    def _gap_boost(self) -> float:
        """根据 Student 与 Teacher 的 mAP 差距，计算额外的 T1 权重提升。"""
        if self._student_map is None or self._teacher_map is None:
            return 0.0
        gap = max(0.0, self._teacher_map - self._student_map)
        # gap 每差 10 个 mAP 点，补偿 0.1 权重，最大 gap_boost_max
        boost = min(self.gap_boost_max, gap / 100.0)
        return boost

    def _build_weight_dict(
        self, base_t1: float, base_t2: float
    ) -> Dict[str, float]:
        """将 base_t1 / base_t2 拆解为各子任务权重。"""
        # return dict(
        #     # ---- Teacher-1 (Online GLIP) ----
        #     # 软标签蒸馏：分类 / 回归 / 中心度
        #     t1_soft_cls   = base_t1 * 1.5,   # 语义分类最重要
        #     t1_soft_reg   = base_t1 * 1.0,
        #     t1_soft_ctr   = base_t1 * 0.8,
        #     # 特征对齐（FPN 多尺度 cosine loss）
        #     t1_feat       = base_t1 * 0.5,
        #     # 文本-视觉响应图对齐（跨模态能力迁移）
        #     t1_text_align = base_t1 * 0.6,
        #     # ---- Teacher-2 (Pseudo Label) ----
        #     # quality-score 加权的 hard-label 检测 loss
        #     t2_pseudo_det = base_t2 * 1.2,   # 伪标签整体稍重，弥补 hard label 噪声
        #     # ---- Student 自身 ----
        #     det           = 1.0,              # 始终不变
        # )
        return dict(
            # ---- Teacher-1 (Online GLIP) ----
            # 软标签蒸馏：分类 / 回归 / 中心度
            t1_soft_cls   = base_t1 * 1.5,   # 语义分类最重要
            t1_soft_reg   = base_t1 * 1.0,
            # t1_soft_ctr   = base_t1 * 0.8,
            t1_soft_ctr   = base_t1 * 0.2,
            # 特征对齐（FPN 多尺度 cosine loss）
            t1_feat       = base_t1 * 0.5,
            # 文本-视觉响应图对齐（跨模态能力迁移）
            t1_text_align = base_t1 * 0.6,
            # ---- Teacher-2 (Pseudo Label) ----
            # quality-score 加权的 hard-label 检测 loss
            # t2_pseudo_det = base_t2 * 1.2,   # 伪标签整体稍重，弥补 hard label 噪声
            t2_pseudo_det = base_t2 * 0.5,   # 伪标签整体稍重，弥补 hard label 噪声
            # ---- Student 自身 ----
            det           = 1.0,              # 始终不变
        )
