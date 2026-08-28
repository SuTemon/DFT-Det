# Copyright (c) FusionKD. All rights reserved.
import copy
from functools import partial
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType, reduce_mean
from mmengine.structures import InstanceData

from ..task_modules.prior_generators import anchor_inside_flags
from ..utils import images_to_levels, multi_apply, unmap

from .dual_teacher_scheduler import DualTeacherDistillationScheduler,DualTeacherDistillationScheduler_loss_weight
from .fusionkd_atss import FusionKDATSS


@MODELS.register_module()
class DualTeacherATSS_loss_weight(FusionKDATSS):
    """双教师蒸馏检测器。

    Teacher-1 (Online, GLIP/GroundingDINO)
        · 在线前向，提供 soft cls / reg / center / feat / text-align 蒸馏
        · 由 DualTeacherDistillationScheduler 控制权重，三阶段动态衰减

    Teacher-2 (Static Pseudo Teacher, 离线 pseudo_coco.json)
        · gt_instances.pseudo_scores 记录每个 box 的质量权重
        · 复用学生 ATSSHead 计算标准检测 loss，用 pseudo_score 重加权
        · 特征通过缓存机制一次提取，不重复计算

    关键驱动入口：
        · set_epoch()                         ← SetEpochInfoHook，每 epoch 调一次
        · update_teacher_student_performance() ← DualTeacherPerformanceHook，每 val 后调一次
    """

    def __init__(self, kd_cfg: OptConfigType = None, **kwargs) -> None:
        super().__init__(kd_cfg=kd_cfg, **kwargs)

        dual_cfg = {} if kd_cfg is None else kd_cfg.get('dual_teacher_cfg', {})

        # 优先从 dual_teacher_cfg.total_epochs 读取，保证 Scheduler 曲线正确
        total_epochs = (
            dual_cfg.get('total_epochs')
            or (kwargs.get('train_cfg') or {}).get('max_epochs')
            or 100
        )
        self.dual_teacher_scheduler = DualTeacherDistillationScheduler_loss_weight(
            total_epochs=total_epochs,
            warmup_ratio=dual_cfg.get('warmup_ratio', 0.30),
            decay_ratio=dual_cfg.get('decay_ratio', 0.70),
            t1_min_weight=dual_cfg.get('t1_min_weight', 0.20),
            t2_min_weight=dual_cfg.get('t2_min_weight', 0.10),
            gap_boost_max=dual_cfg.get('gap_boost_max', 0.40),
            schedule_mode=dual_cfg.get('schedule_mode', 'cosine'),
        )

        self.current_epoch = 0
        self.enable_dual_teacher = dual_cfg.get('enable', True)
        self.enable_t2_pseudo = dual_cfg.get('enable_t2_pseudo', True)
        self.pseudo_score_reduce = dual_cfg.get('pseudo_score_reduce', 'mean')
        self.pseudo_weight_mode = dual_cfg.get('pseudo_weight_mode', 'anchor')

        # 显式初始化特征缓存，避免子类或异常路径访问未初始化属性
        self._stu_feat_cache = None

        # 冷启动：确保 loss() 在首次 set_epoch 前也能拿到合法权重
        self.dual_teacher_scheduler.step(0)

    # ------------------------------------------------------------------
    # 外部驱动接口（由 Hook 调用）
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """每个 epoch 开始时由 SetEpochInfoHook 唯一调用一次。"""
        self.current_epoch = epoch
        self.dual_teacher_scheduler.step(epoch)

    def update_teacher_student_performance(
        self,
        student_map50: float,
        teacher_map50: Optional[float] = None,
    ) -> None:
        """val 结束后由 DualTeacherPerformanceHook 调用，驱动 gap_boost。"""
        self.dual_teacher_scheduler.update_performance(
            student_map50=student_map50,
            teacher_map50=teacher_map50,
        )

    # ------------------------------------------------------------------
    # extract_feat 缓存拦截
    # ------------------------------------------------------------------

    def extract_feat(self, batch_inputs: Tensor):
        """缓存命中时直接返回，避免父类 loss() 中重复提取特征。"""
        if self._stu_feat_cache is not None:
            return self._stu_feat_cache
        return super().extract_feat(batch_inputs)

    # ------------------------------------------------------------------
    # 核心 loss 入口
    # ------------------------------------------------------------------

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """双教师总 loss。

        流程：
          1. 一次 extract_feat 提取全 batch 学生特征
          2. 注入缓存 → 父类 loss() 命中缓存，不重复计算
          3. try/finally 确保缓存必然清理，防止异常后静默污染
          4. 对 Teacher-1 KD loss 和 Student GT loss 按调度权重重标定
          5. Teacher-2 pseudo loss：从已有特征按 index 切片，不重新前向
          6. 记录调度权重到日志
        """
        weights = self.dual_teacher_scheduler.weights if self.enable_dual_teacher else {}

        # Step 1: 一次性提取全 batch 学生特征
        stu_x_full = super().extract_feat(batch_inputs)

        # Step 2~3: 注入缓存，用 try/finally 保证必然清理
        self._stu_feat_cache = stu_x_full
        try:
            losses = super().loss(batch_inputs, batch_data_samples)
        finally:
            # 无论 super().loss() 是否抛出异常，缓存都必须清理
            self._stu_feat_cache = None

        # Step 4: Teacher-1 KD loss 动态重标定
        t1_map = {
            'loss_cls_kd':    weights.get('t1_soft_cls', 1.0),
            'loss_reg_kd':    weights.get('t1_soft_reg', 1.0),
            'loss_center_kd': weights.get('t1_soft_ctr', 1.0),
        }
        for loss_key, scale in t1_map.items():
            if loss_key in losses:
                losses[loss_key] = self._scale_loss(losses[loss_key], scale)

        # Student GT loss 权重（默认 1.0）
        det_weight = weights.get('det', 1.0)
        for key in ['loss_cls', 'loss_bbox', 'loss_centerness']:
            if key in losses:
                losses[key] = self._scale_loss(losses[key], det_weight)

        # Step 5: Teacher-2 pseudo hard-label loss
        pseudo_losses = self._compute_pseudo_teacher_loss(
            batch_data_samples, stu_x_full, weights)
        losses.update(pseudo_losses)

        # Step 6: 调度权重写入日志（纯监控量，必须 detach 避免被加入总 loss 反向图）
        if self.enable_dual_teacher:
            losses['dual_t1_weight'] = batch_inputs.new_tensor(
                weights.get('t1_soft_cls', 0.0)).detach()
            losses['dual_t2_weight'] = batch_inputs.new_tensor(
                weights.get('t2_pseudo_det', 0.0)).detach()

        return losses

    # ------------------------------------------------------------------
    # Teacher-2 pseudo loss
    # ------------------------------------------------------------------

    def _compute_pseudo_teacher_loss(
        self,
        batch_data_samples: SampleList,
        stu_x_full: tuple,
        weights: dict,
    ) -> Dict[str, Tensor]:
        """Teacher-2 pseudo hard-label 检测 loss。

        特征直接从 stu_x_full 按 pseudo_indices 切片复用，零额外 extract_feat 开销。

        forward_hkd_single 签名为 (self, x, scale, module)，
        用 partial(module=self) 预绑定 module，再通过 multi_apply 并行 map。
        """
        pseudo_indices = self._get_pseudo_indices(batch_data_samples)
        if (not self.enable_t2_pseudo) or (not pseudo_indices):
            return {}

        pseudo_weight = weights.get('t2_pseudo_det', 1.0)
        pseudo_samples = [batch_data_samples[i] for i in pseudo_indices]

        # 从特征取 device，不依赖 batch_inputs
        feat_device = stu_x_full[0].device
        idx_tensor = torch.tensor(pseudo_indices,
                                  dtype=torch.long,
                                  device=feat_device)
        stu_x_pseudo = tuple(feat[idx_tensor] for feat in stu_x_full)

        # forward_hkd_single(self, x, scale, module) — module 是位置参数
        # partial 将 module=self 预绑定为关键字参数，等价于
        #   forward_hkd_single(x, scale, module=self)，合法且已验证
        from ..utils import multi_apply
        cls_scores, bbox_preds, centernesses, _, _ = multi_apply(
            partial(self.forward_hkd_single, module=self),
            stu_x_pseudo,
            self.bbox_head.scales,
        )

        # Teacher-2 支持两种模式：
        #   anchor: 当前主版本，按正样本 anchor 的 pseudo_score 细粒度加权
        #   sample: 对整路 pseudo loss 乘一个 batch-level pseudo_score 标量，用于消融对照
        if self.pseudo_weight_mode == 'sample':
            sample_level_losses = self._loss_by_feat_pseudo_sample_weighted(
                cls_scores,
                bbox_preds,
                centernesses,
                pseudo_samples,
                pseudo_weight,
            )
            pseudo_pos_score_mean = sample_level_losses.pop('pseudo_pos_score_mean').detach()
            result = {f'{key}_pseudo': value for key, value in sample_level_losses.items()}
            result['pseudo_pos_score_mean'] = pseudo_pos_score_mean
            return result

        anchor_level_losses = self._loss_by_feat_pseudo_anchor_weighted(
            cls_scores,
            bbox_preds,
            centernesses,
            pseudo_samples,
            pseudo_weight,
        )

        # pseudo_pos_score_mean 是纯监控量，单独 detach 后命名，不参与 backward
        pseudo_pos_score_mean = anchor_level_losses.pop('pseudo_pos_score_mean').detach()

        result = {f'{key}_pseudo': value for key, value in anchor_level_losses.items()}
        result['pseudo_pos_score_mean'] = pseudo_pos_score_mean
        return result

    def _loss_by_feat_pseudo_sample_weighted(
        self,
        cls_scores,
        bbox_preds,
        centernesses,
        batch_data_samples,
        pseudo_weight: float,
    ) -> Dict[str, Tensor]:
        """Teacher-2 的 sample-level pseudo 加权损失。

        用于消融：先按标准 ATSS 计算 pseudo det loss，
        再用整个 pseudo batch 的 pseudo_score 标量统一缩放三路损失。
        """
        from ..utils import unpack_gt_instances

        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = unpack_gt_instances(batch_data_samples)

        raw_losses = self.bbox_head.loss_by_feat(
            cls_scores,
            bbox_preds,
            centernesses,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
        )

        pseudo_pos_score_mean = self._reduce_batch_pseudo_score(batch_data_samples)
        final_scale = pseudo_pos_score_mean * pseudo_weight

        return dict(
            loss_cls=self._scale_loss(raw_losses['loss_cls'], final_scale),
            loss_bbox=self._scale_loss(raw_losses['loss_bbox'], final_scale),
            loss_centerness=self._scale_loss(raw_losses['loss_centerness'], final_scale),
            pseudo_pos_score_mean=pseudo_pos_score_mean,
        )

    def _loss_by_feat_pseudo_anchor_weighted(
        self,
        cls_scores,
        bbox_preds,
        centernesses,
        batch_data_samples,
        pseudo_weight: float,
    ) -> Dict[str, Tensor]:
        """Teacher-2 的 anchor 级 pseudo_score 加权损失。

        这里有两类正交权重：
          1. pseudo_score：来自每个 pseudo GT 的质量分数，细粒度映射到正样本 anchor
          2. pseudo_weight：来自 Scheduler 的阶段性调度权重，控制整路 T2 loss 的全局强度

        核心思想：
          1. 使用 ATSS assigner/sampler 正常完成 anchor→GT 分配
          2. 对每个正样本 anchor，找到其对应的 GT 索引 pos_assigned_gt_inds
          3. 用该 GT 的 pseudo_score 作为该 anchor 的质量权重
          4. cls / bbox / centerness 都使用 anchor-level 权重，而非整 batch 标量
        """
        from ..utils import unpack_gt_instances

        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = unpack_gt_instances(batch_data_samples)

        featmap_sizes = [featmap.size()[-2:] for featmap in bbox_preds]
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.bbox_head.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = self._get_pseudo_targets(
            anchor_list=anchor_list,
            valid_flag_list=valid_flag_list,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            unmap_outputs=True,
        )

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pseudo_label_weights_list,
         pseudo_bbox_weights_list, avg_factor) = cls_reg_targets

        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=device)).item()

        losses_cls, losses_bbox, losses_centerness = multi_apply(
            self._pseudo_loss_by_feat_single,
            anchor_list,
            cls_scores,
            bbox_preds,
            centernesses,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            pseudo_label_weights_list,
            pseudo_bbox_weights_list,
            avg_factor=avg_factor,
        )

        # 统计所有层正样本 anchor 的平均 pseudo_score，用于训练日志监控
        # pseudo_bbox_weights_list[level] shape: (num_imgs, num_anchors_per_level)
        # 值域：正样本 = pseudo_score ∈ (0,1]，其余 anchor = 0
        all_pos_scores = torch.cat([
            pw.reshape(-1) for pw in pseudo_bbox_weights_list
        ])
        pos_mask = all_pos_scores > 0
        pseudo_pos_score_mean = (
            all_pos_scores[pos_mask].mean()
            if pos_mask.any()
            else all_pos_scores.new_tensor(0.0)
        )

        return dict(
            loss_cls=self._scale_loss(losses_cls, pseudo_weight),
            loss_bbox=self._scale_loss(losses_bbox, pseudo_weight),
            loss_centerness=self._scale_loss(losses_centerness, pseudo_weight),
            pseudo_pos_score_mean=pseudo_pos_score_mean,
        )

    def _get_pseudo_targets(
        self,
        anchor_list,
        valid_flag_list,
        batch_gt_instances,
        batch_img_metas,
        batch_gt_instances_ignore=None,
        unmap_outputs=True,
    ):
        """在 ATSS 标准 target 基础上，额外生成 anchor-level pseudo 权重。"""
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # 避免原地修改 get_anchors() 返回结果，污染同一 forward 中的其他调用方
        anchor_list = copy.deepcopy(anchor_list)
        valid_flag_list = copy.deepcopy(valid_flag_list)

        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        for i in range(num_imgs):
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs

        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, all_pseudo_label_weights, all_pseudo_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = multi_apply(
            self._get_pseudo_targets_single,
            anchor_list,
            valid_flag_list,
            num_level_anchors_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            unmap_outputs=unmap_outputs,
        )

        avg_factor = sum([results.avg_factor for results in sampling_results_list])
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
        pseudo_label_weights_list = images_to_levels(all_pseudo_label_weights, num_level_anchors)
        pseudo_bbox_weights_list = images_to_levels(all_pseudo_bbox_weights, num_level_anchors)

        return (
            anchors_list, labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, pseudo_label_weights_list,
            pseudo_bbox_weights_list, avg_factor
        )

    def _get_pseudo_targets_single(
        self,
        flat_anchors: Tensor,
        valid_flags: Tensor,
        num_level_anchors: List[int],
        gt_instances: InstanceData,
        img_meta: dict,
        gt_instances_ignore: Optional[InstanceData] = None,
        unmap_outputs: bool = True,
    ):
        """单图 ATSS target + anchor-level pseudo 权重生成。"""
        inside_flags = anchor_inside_flags(
            flat_anchors, valid_flags, img_meta['img_shape'][:2],
            self.bbox_head.train_cfg['allowed_border'])
        if not inside_flags.any():
            raise ValueError(
                'There is no valid anchor inside the image boundary. Please '
                'check the image size and anchor sizes, or set '
                'allowed_border to -1 to skip the condition.')

        anchors = flat_anchors[inside_flags, :]
        num_level_anchors_inside = self.bbox_head.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)

        pred_instances = InstanceData(priors=anchors)
        assign_result = self.bbox_head.assigner.assign(
            pred_instances, num_level_anchors_inside,
            gt_instances, gt_instances_ignore)
        sampling_result = self.bbox_head.sampler.sample(
            assign_result, pred_instances, gt_instances)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,), self.bbox_head.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        # 新增：anchor-level pseudo 权重
        pseudo_label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        pseudo_bbox_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        if len(pos_inds) > 0:
            if self.bbox_head.reg_decoded_bbox:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            else:
                pos_bbox_targets = self.bbox_head.bbox_coder.encode(
                    sampling_result.pos_priors, sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            labels[pos_inds] = sampling_result.pos_gt_labels
            label_weights[pos_inds] = 1.0 if self.bbox_head.train_cfg['pos_weight'] <= 0 \
                else self.bbox_head.train_cfg['pos_weight']

            # 核心：从 GT pseudo_scores 映射到正样本 anchor
            if hasattr(gt_instances, 'pseudo_scores') and len(gt_instances.pseudo_scores) > 0:
                gt_pseudo_scores = gt_instances.pseudo_scores.float().to(anchors.device)
                pos_gt_inds = sampling_result.pos_assigned_gt_inds.long()
                pos_pseudo_scores = gt_pseudo_scores[pos_gt_inds]
                pseudo_label_weights[pos_inds] = pos_pseudo_scores
                pseudo_bbox_weights[pos_inds] = pos_pseudo_scores
            else:
                pseudo_label_weights[pos_inds] = 1.0
                pseudo_bbox_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.bbox_head.num_classes)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            pseudo_label_weights = unmap(pseudo_label_weights, num_total_anchors, inside_flags)
            pseudo_bbox_weights = unmap(pseudo_bbox_weights, num_total_anchors, inside_flags)

        return (
            anchors, labels, label_weights, bbox_targets, bbox_weights,
            pseudo_label_weights, pseudo_bbox_weights,
            pos_inds, neg_inds, sampling_result
        )

    def _pseudo_loss_by_feat_single(
        self,
        anchors: Tensor,
        cls_score: Tensor,
        bbox_pred: Tensor,
        centerness: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        pseudo_label_weights: Tensor,
        pseudo_bbox_weights: Tensor,
        avg_factor: float,
    ):
        """ATSS 单层 anchor-level pseudo-score 加权损失。"""
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.bbox_head.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        pseudo_label_weights = pseudo_label_weights.reshape(-1)
        pseudo_bbox_weights = pseudo_bbox_weights.reshape(-1)

        # 分类：只对正样本乘 pseudo_score，负样本保持原始 label_weights
        final_label_weights = label_weights.clone()
        bg_class_ind = self.bbox_head.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        if len(pos_inds) > 0:
            final_label_weights[pos_inds] *= pseudo_label_weights[pos_inds]

        loss_cls = self.bbox_head.loss_cls(
            cls_score, labels, final_label_weights, avg_factor=avg_factor)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]
            pos_pseudo_bbox_weights = pseudo_bbox_weights[pos_inds]

            centerness_targets = self.bbox_head.centerness_target(pos_anchors, pos_bbox_targets)
            pos_decode_bbox_pred = self.bbox_head.bbox_coder.decode(pos_anchors, pos_bbox_pred)

            # 回归：使用 anchor-level pseudo_score 做质量加权，
            # avg_factor 直接复用标准 ATSS 的正样本归一化因子
            loss_bbox = self.bbox_head.loss_bbox(
                pos_decode_bbox_pred,
                pos_bbox_targets,
                weight=centerness_targets * pos_pseudo_bbox_weights,
                avg_factor=avg_factor)

            # centerness：anchor 级 pseudo_score 加权
            loss_centerness = self.bbox_head.loss_centerness(
                pos_centerness,
                centerness_targets,
                weight=pos_pseudo_bbox_weights,
                avg_factor=avg_factor)
        else:
            loss_bbox = cls_score.sum() * 0
            loss_centerness = cls_score.sum() * 0

        return loss_cls, loss_bbox, loss_centerness

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _get_pseudo_indices(self, batch_data_samples: SampleList) -> List[int]:
        return [
            i for i, ds in enumerate(batch_data_samples)
            if hasattr(ds.gt_instances, 'pseudo_scores')
        ]

    def _reduce_batch_pseudo_score(self,
                                   batch_data_samples: SampleList) -> Tensor:
        """将 pseudo 样本内所有 GT 的 pseudo_scores 归约为 batch-level 标量。"""
        score_tensors = []
        for ds in batch_data_samples:
            if hasattr(ds.gt_instances, 'pseudo_scores') and len(ds.gt_instances.pseudo_scores) > 0:
                score_tensors.append(ds.gt_instances.pseudo_scores.float())

        if not score_tensors:
            return next(self.parameters()).new_tensor(1.0)

        all_scores = torch.cat(score_tensors)
        if self.pseudo_score_reduce == 'max':
            return all_scores.max()
        if self.pseudo_score_reduce == 'sum':
            return all_scores.sum() / max(float(all_scores.numel()), 1.0)
        return all_scores.mean()

    @staticmethod
    def _scale_loss(loss_value, scale):
        if isinstance(loss_value, list):
            return [item * scale for item in loss_value]
        return loss_value * scale
