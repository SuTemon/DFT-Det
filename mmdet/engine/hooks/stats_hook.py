import time
from collections import OrderedDict
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from mmengine.analysis import get_model_complexity_info
from mmengine.hooks import Hook
from mmengine.logging import MMLogger
from mmengine.model import BaseModel, is_model_wrapper, revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

from mmdet.registry import HOOKS


def _fmt_count(num: int) -> str:
    if num >= 1e9:
        return f'{num / 1e9:.3f} B'
    if num >= 1e6:
        return f'{num / 1e6:.3f} M'
    if num >= 1e3:
        return f'{num / 1e3:.3f} K'
    return str(num)


def _fmt_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f'{hours}:{minutes:02d}:{secs:02d}'


def _iter_unique_parameters(module: nn.Module, prefix: str = '') -> Iterable[Tuple[str, nn.Parameter]]:
    seen = set()
    for name, param in module.named_parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        yield f'{prefix}{name}', param


def _count_parameters(module: nn.Module) -> Dict[str, int]:
    total = 0
    trainable = 0
    for _, param in _iter_unique_parameters(module):
        numel = param.numel()
        total += numel
        if param.requires_grad:
            trainable += numel
    return dict(total=total, trainable=trainable, frozen=total - trainable)


def _get_teacher_module(model: nn.Module) -> Optional[nn.Module]:
    teacher = getattr(model, 'teacher', None)
    return teacher if isinstance(teacher, nn.Module) else None


def _get_top_level_parts(model: nn.Module) -> OrderedDict:
    parts = OrderedDict()
    for name, child in model.named_children():
        parts[name] = child
    teacher = _get_teacher_module(model)
    if teacher is not None and 'teacher' not in parts:
        parts['teacher'] = teacher
    return parts


def _get_second_level_part_stats(model: nn.Module) -> OrderedDict:
    target_parts = ('backbone', 'neck', 'bbox_head', 'teacher')
    second_level_stats = OrderedDict()

    for part_name in target_parts:
        if part_name == 'teacher':
            part_module = _get_teacher_module(model)
        else:
            part_module = getattr(model, part_name, None)

        if not isinstance(part_module, nn.Module):
            continue

        child_stats = OrderedDict()
        for child_name, child_module in part_module.named_children():
            child_stats[child_name] = _count_parameters(child_module)
        second_level_stats[part_name] = child_stats

    return second_level_stats


def _append_stats_table(lines, title: str, stats_dict: OrderedDict, indent: str = '') -> None:
    if not stats_dict:
        return

    lines.extend([
        title,
        f"{indent}{'模块':<28}{'总量':>16}{'可训练':>16}{'冻结':>16}",
        f"{indent}{'-' * 76}",
    ])
    for name, stats in stats_dict.items():
        lines.append(
            f"{indent}{name:<28}{_fmt_count(stats['total']):>16}{_fmt_count(stats['trainable']):>16}{_fmt_count(stats['frozen']):>16}"
        )


def _log_param_stats(model: nn.Module, logger: MMLogger) -> None:
    student_stats = _count_parameters(model)
    teacher = _get_teacher_module(model)
    teacher_stats = _count_parameters(teacher) if teacher is not None else None

    total = student_stats['total'] + (teacher_stats['total'] if teacher_stats else 0)
    trainable = student_stats['trainable'] + (teacher_stats['trainable'] if teacher_stats else 0)
    frozen = total - trainable

    lines = [
        '',
        '=' * 88,
        'FusionKD 参数统计',
        '=' * 88,
        f"{'总参数量':<24}: {_fmt_count(total):>12} ({total:,})",
        f"{'可训练参数量':<24}: {_fmt_count(trainable):>12} ({trainable:,})",
        f"{'冻结参数量':<24}: {_fmt_count(frozen):>12} ({frozen:,})",
        f"{'可训练占比':<24}: {((trainable / total) * 100 if total else 0.0):>11.2f}%",
        '-' * 88,
    ]

    part_stats = OrderedDict()
    part_stats['student(all registered)'] = student_stats
    if teacher_stats is not None:
        part_stats['teacher'] = teacher_stats
    for name, child in _get_top_level_parts(model).items():
        part_stats[name] = _count_parameters(child)

    _append_stats_table(lines, '一级模块参数量', part_stats)

    second_level_stats = _get_second_level_part_stats(model)
    for part_name, child_stats in second_level_stats.items():
        lines.extend([
            '-' * 88,
            f'{part_name} 二级模块参数量',
        ])
        if child_stats:
            _append_stats_table(lines, '', child_stats, indent='  ')
        else:
            lines.append('  (无可展开子模块)')

    lines.append('=' * 88)
    logger.info('\n'.join(lines))


class _ComplexityModelAdapter(nn.Module):
    def __init__(self, model: BaseModel):
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor):
        return self.model(inputs, mode='tensor')


@HOOKS.register_module()
class FusionKDStatsHook(Hook):
    priority = 'LOW'

    def __init__(
        self,
        warmup_iters: int = 10,
        measure_iters: int = 50,
        compute_fps: bool = True,
        compute_latency: bool = True,
        compute_complexity: bool = True,
    ) -> None:
        self.warmup_iters = warmup_iters
        self.measure_iters = measure_iters
        self.compute_fps = compute_fps
        self.compute_latency = compute_latency
        self.compute_complexity = compute_complexity

        self._train_start_time: Optional[float] = None
        self._epoch_start_time: Optional[float] = None
        self._epoch_durations = []
        self._complexity_cache = {}

    def before_run(self, runner: Runner) -> None:
        logger = runner.logger
        _log_param_stats(self._unwrap_model(runner.model), logger)
        self._train_start_time = time.perf_counter()
        logger.info('=' * 60)
        logger.info(f'训练开始，总轮数: {runner.max_epochs}')
        logger.info('=' * 60)

    def before_train_epoch(self, runner: Runner) -> None:
        self._epoch_start_time = time.perf_counter()

    def after_train_epoch(self, runner: Runner) -> None:
        if self._epoch_start_time is None or self._train_start_time is None:
            return

        epoch_duration = time.perf_counter() - self._epoch_start_time
        total_duration = time.perf_counter() - self._train_start_time
        self._epoch_durations.append(epoch_duration)

        avg_epoch_duration = sum(self._epoch_durations) / len(self._epoch_durations)
        remaining_epochs = runner.max_epochs - (runner.epoch + 1)
        remaining_duration = avg_epoch_duration * max(remaining_epochs, 0)

        runner.logger.info(
            f'[Epoch {runner.epoch + 1}/{runner.max_epochs}] '
            f'本轮训练用时: {_fmt_seconds(epoch_duration)} | '
            f'累计用时: {_fmt_seconds(total_duration)} | '
            f'预计剩余: {_fmt_seconds(remaining_duration)}'
        )

    def after_val_epoch(self, runner: Runner, metrics: Optional[dict] = None) -> None:
        del metrics
        logger = runner.logger
        model = self._unwrap_model(runner.model)
        data = self._get_val_batch(runner)
        if data is None:
            logger.warning('未能获取验证样本，跳过复杂度/FPS/延迟统计。')
            return

        complexity, fps, latency, errors = self._measure_validation_runtime(
            model, data, runner)

        lines = [
            '',
            '-' * 72,
            f'第 {runner.epoch + 1} 轮验证附加统计',
            '-' * 72,
        ]
        if complexity is not None:
            lines.extend([
                f"{'GFLOPS':<20}: {complexity['gflops']:.4f}",
                f"{'FLOPs(raw)':<20}: {complexity['flops']}",
                f"{'Complexity Params':<20}: {complexity['params']}",
                f"{'输入尺寸':<20}: {complexity['input_shape']}",
                f"{'Pad 后尺寸':<20}: {complexity['pad_shape']}",
            ])
        else:
            lines.append(f"{'GFLOPS':<20}: 统计失败")

        if fps is not None:
            lines.append(f"{'FPS':<20}: {fps:.2f} img/s")
        else:
            lines.append(f"{'FPS':<20}: 统计失败")

        if latency is not None:
            lines.append(f"{'平均延迟':<20}: {latency:.2f} ms/img")
        else:
            lines.append(f"{'平均延迟':<20}: 统计失败")

        if errors:
            lines.append('-' * 72)
            for item in errors:
                lines.append(f"失败原因: {item}")

        lines.append('-' * 72)
        logger.info('\n'.join(lines))

    def after_run(self, runner: Runner) -> None:
        if self._train_start_time is None:
            return
        total_duration = time.perf_counter() - self._train_start_time
        avg_epoch_duration = (
            sum(self._epoch_durations) / len(self._epoch_durations)
            if self._epoch_durations else 0.0
        )
        runner.logger.info('=' * 60)
        runner.logger.info(f'训练结束，总用时: {_fmt_seconds(total_duration)}')
        runner.logger.info(f'平均每轮用时: {_fmt_seconds(avg_epoch_duration)}')
        runner.logger.info('=' * 60)

    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        return model.module if is_model_wrapper(model) else model

    def _get_val_batch(self, runner: Runner) -> Optional[dict]:
        val_loop = getattr(runner, 'val_loop', None)
        if val_loop is None or getattr(val_loop, 'dataloader', None) is None:
            return None
        try:
            return next(iter(val_loop.dataloader))
        except Exception as exc:
            runner.logger.warning(f'读取验证数据失败，跳过统计: {exc}')
            return None

    def _measure_validation_runtime(self, model: nn.Module, data: dict,
                                    runner: Runner):
        model_device = next(model.parameters()).device
        was_training = model.training
        complexity = None
        fps = None
        latency = None
        errors = []

        try:
            with torch.no_grad():
                processed = model.data_preprocessor(data, False)
            inputs = processed['inputs']
            data_samples = processed.get('data_samples', None)
            cache_key = tuple(inputs.shape)

            model.eval()

            if self.compute_complexity:
                try:
                    complexity = self._compute_complexity(model, inputs,
                                                          data_samples,
                                                          cache_key, runner)
                except Exception as exc:
                    errors.append(f'FLOPs 统计失败: {exc}')

            if self.compute_fps or self.compute_latency:
                try:
                    fps, latency = self._measure_fps_and_latency(
                        model, inputs, data_samples, model_device)
                except Exception as exc:
                    errors.append(f'FPS/延迟统计失败: {exc}')
        except Exception as exc:
            errors.append(f'验证统计预处理失败: {exc}')
        finally:
            if was_training:
                model.train()
        return complexity, fps, latency, errors

    def _compute_complexity(self, model: nn.Module, inputs: torch.Tensor,
                            data_samples, cache_key, runner: Runner):
        del data_samples
        if cache_key in self._complexity_cache:
            return self._complexity_cache[cache_key]

        init_default_scope(runner.cfg.get('default_scope', 'mmdet'))
        analysis_model = revert_sync_batchnorm(model)
        adapter = _ComplexityModelAdapter(analysis_model)
        outputs = get_model_complexity_info(
            adapter,
            None,
            inputs=inputs,
            show_table=False,
            show_arch=False)
        result = dict(
            gflops=float(outputs['flops']) / 1e9,
            flops=outputs['flops'],
            params=outputs['params'],
            input_shape=tuple(inputs.shape[-2:]),
            pad_shape=tuple(inputs.shape[-2:]),
        )
        self._complexity_cache[cache_key] = result
        return result

    def _measure_fps_and_latency(self, model: nn.Module, inputs: torch.Tensor,
                                 data_samples, device: torch.device):
        warmup_iters = max(int(self.warmup_iters), 0)
        measure_iters = max(int(self.measure_iters), 0)
        if measure_iters == 0:
            return None, None

        use_cuda = device.type == 'cuda' and torch.cuda.is_available()

        with torch.no_grad():
            for _ in range(warmup_iters):
                model(inputs, data_samples, mode='predict')
            if use_cuda:
                torch.cuda.synchronize(device)

            if use_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(measure_iters):
                    model(inputs, data_samples, mode='predict')
                end.record()
                torch.cuda.synchronize(device)
                total_ms = start.elapsed_time(end)
            else:
                start_time = time.perf_counter()
                for _ in range(measure_iters):
                    model(inputs, data_samples, mode='predict')
                total_ms = (time.perf_counter() - start_time) * 1000

        batch_size = int(inputs.shape[0]) if inputs.ndim > 0 else 1
        avg_latency = total_ms / measure_iters / batch_size
        fps = 1000.0 / avg_latency if avg_latency > 0 else None

        if not self.compute_fps:
            fps = None
        if not self.compute_latency:
            avg_latency = None
        return fps, avg_latency
