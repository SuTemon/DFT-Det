# Copyright (c) FusionKD. All rights reserved.
"""
DualTeacherPerformanceHook
===========================
在每次 val 结束后，把 Student 的 mAP50 注入 DualTeacherDistillationScheduler，
驱动 gap_boost 自适应权重调整。

配置示例
--------
custom_hooks = [
    dict(
        type='DualTeacherPerformanceHook',
        teacher_map50=66.0,              # GLIP 在 val 集上的实测 mAP50（0~100 scale）
        metric_key='coco/bbox_mAP_50',  # CocoMetric 输出的 key（不含 val/ 前缀）
    )
]

metric_key 说明
---------------
MMEngine CocoMetric 评估后的 key 格式取决于版本和配置：
  - log_scalars 中通常带 'val/' 前缀：'val/coco/bbox_mAP_50'
  - runner.message_hub.get_info('metrics') 中通常不带前缀：'coco/bbox_mAP_50'
本实现同时覆盖两种格式，取先命中的值。
"""

from mmengine.hooks import Hook
from mmengine.model.wrappers import is_model_wrapper

from mmdet.registry import HOOKS


@HOOKS.register_module()
class DualTeacherPerformanceHook(Hook):
    """val 后将 Student mAP50 注入双教师调度器，驱动 gap_boost。

    Args:
        teacher_map50 (float | None): Teacher-1 在 val 集上的参考 mAP50
            （0~100 scale），填写离线实测值。若为 None，gap_boost 不生效。
        metric_key (str): CocoMetric 输出的指标 key，不含 'val/' 前缀，
            默认 'coco/bbox_mAP_50'。
    """

    priority = 'LOW'

    def __init__(
        self,
        teacher_map50: float = None,
        metric_key: str = 'coco/bbox_mAP_50',
    ) -> None:
        self.teacher_map50 = teacher_map50
        self.metric_key = metric_key

    def after_val_epoch(self, runner, metrics: dict = None) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        if not hasattr(model, 'update_teacher_student_performance'):
            return

        student_map50 = self._extract_map50(runner, metrics)
        if student_map50 is None:
            runner.logger.warning(
                f'[DualTeacherPerformanceHook] 未能从 metrics 或 '
                f'message_hub 中读取到 key="{self.metric_key}"，'
                f'本轮跳过 gap_boost 更新。')
            return

        model.update_teacher_student_performance(
            student_map50=student_map50,
            teacher_map50=self.teacher_map50,
        )

        # 安全地调用 log_info，若未实现则 fallback 到权重字典
        scheduler = model.dual_teacher_scheduler
        if hasattr(scheduler, 'log_info'):
            log_str = scheduler.log_info()
        else:
            log_str = str(getattr(scheduler, 'weights', {}))

        runner.logger.info(
            f'[DualTeacher] epoch={runner.epoch} '
            f'student_mAP50={student_map50:.2f} '
            f'teacher_mAP50={self.teacher_map50} '
            f'| {log_str}')

    def _to_map50_scale(self, val) -> float:
        """Convert metric scalar to 0~100 mAP50 scale."""
        raw = float(val)
        return raw if raw > 1.0 else raw * 100.0

    def _read_scalar(self, val):
        """Read numeric value from scalar or HistoryBuffer."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return val
        current = getattr(val, 'current', None)
        if callable(current):
            try:
                return current()
            except ValueError:
                pass
        mean = getattr(val, 'mean', None)
        if callable(mean):
            return mean()
        return val

    def _extract_map50(self, runner, metrics) -> float:
        """从多个来源依次尝试读取 student mAP50，返回 float 或 None。

        优先级：
          1. 传入的 metrics dict（MMEngine val_loop 会传）
          2. runner.message_hub.get_info('metrics')（标准接口，无前缀 key）
          3. runner.message_hub.log_scalars（精确匹配 'val/<metric_key>'）
        """
        # --- 路径 1: metrics 参数 ---
        if metrics is not None:
            val = metrics.get(self.metric_key)
            if val is not None:
                try:
                    return self._to_map50_scale(self._read_scalar(val))
                except (TypeError, ValueError):
                    pass

        # --- 路径 2: message_hub.get_info('metrics')（标准接口，值域 0~1）---
        try:
            hub_metrics = runner.message_hub.get_info('metrics')
            if isinstance(hub_metrics, dict):
                val = hub_metrics.get(self.metric_key)
                if val is not None:
                    return self._to_map50_scale(self._read_scalar(val))
        except (KeyError, AttributeError, TypeError, ValueError):
            pass

        # --- 路径 3: log_scalars 精确 key 匹配（带 val/ 前缀）---
        try:
            val_log = runner.message_hub.log_scalars
            exact_key = f'val/{self.metric_key}'
            if exact_key in val_log:
                return self._to_map50_scale(self._read_scalar(val_log[exact_key]))
        except (AttributeError, KeyError, TypeError, ValueError):
            pass

        return None
