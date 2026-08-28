# Copyright (c) FusionKD. All rights reserved.
import copy
import math
from functools import partial
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptConfigType, OptInstanceList, reduce_mean
from mmengine.structures import InstanceData

from ..task_modules.prior_generators import anchor_inside_flags
from ..utils import images_to_levels, multi_apply, unmap

from .dual_teacher_scheduler import DualTeacherDistillationScheduler
from .dual_teacher_scheduler import DualTeacherDistillationScheduler_loss_weight
from .fusionkd_atss import FusionKDATSS


@MODELS.register_module()
class DualTeacherATSS(FusionKDATSS):
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

    def __init__(self,
                 kd_cfg: OptConfigType = None,
                 sgda_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super().__init__(kd_cfg=kd_cfg, **kwargs)

        dual_cfg = {} if kd_cfg is None else kd_cfg.get('dual_teacher_cfg', {})

        # 优先从 dual_teacher_cfg.total_epochs 读取，保证 Scheduler 曲线正确
        total_epochs = (
            dual_cfg.get('total_epochs')
            or (kwargs.get('train_cfg') or {}).get('max_epochs')
            or 100
        )
        self.dual_teacher_scheduler = DualTeacherDistillationScheduler(
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
        self.off_bc_cfg = dict(
            enable=False,
            gamma=2.0,
            delta=2.0,
            lambda_ent=0.05,
            detach_modulator=False,
            reg_score_floor=0.5,
            scale_entropy_by_scheduler=False,
            pseudo_neg_weight=0.0,
        )
        self.off_bc_cfg.update(dual_cfg.get('off_bc_cfg', {}))

        self.pacs_cfg = dict(
            enable=False,
            in_channels=256,
            hidden_channels=64,
            w_min=0.2,
            warmup_epochs=10,
            lb_loss_weight=0.01,
            detach_router_input=True,
        )
        self.pacs_cfg.update(dual_cfg.get('pacs_cfg', {}))
        if self.pacs_cfg.get('enable', False) and self.pseudo_weight_mode != 'anchor':
            raise ValueError('PaCS Phase-1 requires pseudo_weight_mode="anchor".')

        if self.pacs_cfg.get('enable', False):
            from .pacs_router import PacsRouter
            self._assert_pacs_anchor_compatible()
            self.pacs_router = PacsRouter(
                in_channels=self.pacs_cfg.get('in_channels', 256),
                hidden_channels=self.pacs_cfg.get('hidden_channels', 64),
                num_branches=1,
                detach_input=self.pacs_cfg.get('detach_router_input', True),
            )
            self._pacs_lambda_t2_max = self._precompute_pacs_lambda_max(
                key='t2_pseudo_det')
        else:
            self.pacs_router = None
            self._pacs_lambda_t2_max = 1.0

        if sgda_cfg is not None:
            self.sgda_loss = MODELS.build(sgda_cfg)
            self.use_sgda = True
        else:
            self.sgda_loss = None
            self.use_sgda = False

        # 显式初始化特征缓存，避免子类或异常路径访问未初始化属性
        self._stu_feat_cache = None
        self._off_bc_iter = 0

        # 冷启动：确保 loss() 在首次 set_epoch 前也能拿到合法权重
        self.dual_teacher_scheduler.step(0)

    def _assert_pacs_anchor_compatible(self) -> None:
        num_base_priors = self.bbox_head.prior_generator.num_base_priors
        if isinstance(num_base_priors, (list, tuple)):
            valid = all(int(item) == 1 for item in num_base_priors)
        else:
            valid = int(num_base_priors) == 1
        if not valid:
            raise ValueError(
                'PaCS Phase-1 assumes one anchor per spatial location. '
                f'Got num_base_priors={num_base_priors}.')

    def _precompute_pacs_lambda_max(self, key: str) -> float:
        scheduler = self.dual_teacher_scheduler
        backup_epoch = scheduler.current_epoch
        backup_weights = copy.deepcopy(scheduler.weights)
        backup_stage = scheduler.stage

        max_weight = 0.0
        try:
            for epoch in range(scheduler.total_epochs):
                scheduler.step(epoch)
                max_weight = max(
                    max_weight, float(scheduler.weights.get(key, 0.0)))
        finally:
            scheduler.current_epoch = backup_epoch
            scheduler._weights = backup_weights
            scheduler._stage = backup_stage

        return max(max_weight, 1e-3)

    def _pacs_warmup_eta(self, device=None) -> Tensor:
        warmup_epochs = int(self.pacs_cfg.get('warmup_epochs', 0))
        if warmup_epochs > 0 and self.current_epoch < warmup_epochs:
            eta = float(self.current_epoch) / max(float(warmup_epochs), 1.0)
        else:
            eta = 1.0
        if device is None:
            return torch.tensor(eta)
        return torch.tensor(eta, device=device)

    def _compute_pacs_t2_weight(self, stu_feat: Tensor):
        spatial_logit = self.pacs_router(stu_feat)
        batch_size, _, height, width = spatial_logit.shape
        spatial_logit = spatial_logit.reshape(batch_size, height * width)

        lambda_t2 = float(
            self.dual_teacher_scheduler.weights.get('t2_pseudo_det', 1.0))
        rho_t2 = lambda_t2 / max(float(self._pacs_lambda_t2_max), 1e-3)
        rho_t2 = max(0.05, min(0.95, rho_t2))
        phi_t2 = math.log(rho_t2 / (1.0 - rho_t2))

        w2_raw = torch.sigmoid(spatial_logit + phi_t2)
        w_min = float(self.pacs_cfg.get('w_min', 0.2))
        w2_lb = w_min + (1.0 - w_min) * w2_raw
        eta = self._pacs_warmup_eta(device=w2_raw.device)
        w2_eff = (1.0 - eta) + eta * w2_lb
        phi_sigmoid = w2_raw.new_tensor(rho_t2)
        return w2_eff, w2_raw, phi_sigmoid, eta

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

    def loss_by_feat(
        self,
        tea_cls_scores: List[Tensor],
        tea_bbox_preds: List[Tensor],
        tea_centernesses: List[Tensor],
        tea_feats: List[Tensor],
        tea_reg_hold,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        centernesses: List[Tensor],
        feats: List[Tensor],
        fused_feats,
        t_text,
        s_text, 
        reused_cls_scores,
        reused_bbox_preds,
        reused_centernesses,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> dict:
        """Append SGDA to the original FusionKD feature losses."""
        raw_tea_cls_scores = tea_cls_scores
        losses = super().loss_by_feat(
            tea_cls_scores,
            tea_bbox_preds,
            tea_centernesses,
            tea_feats,
            tea_reg_hold,
            cls_scores,
            bbox_preds,
            centernesses,
            feats,
            fused_feats,
            t_text,
            s_text,
            reused_cls_scores,
            reused_bbox_preds,
            reused_centernesses,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
        )

        if self.use_sgda:
            #先调用 _build_fg_masks_for_sgda，传入教师的分类预测（raw_tea_cls_scores）和特征（tea_reg_hold），动态计算出每个特征层的前景权重。
            fg_masks = self._build_fg_masks_for_sgda(
                raw_tea_cls_scores, tea_reg_hold)
            sgda_losses = self.sgda_loss(
                feats_s=fused_feats,
                feats_t=tea_reg_hold,
                fg_masks=fg_masks,
            )
            losses.update(sgda_losses)

        return losses

    def _build_fg_masks_for_sgda(self, tea_cls_scores, tea_reg_hold):
        """Build foreground reliability masks for SGDA.

        GLIP token logits are not necessarily calibrated so that foreground
        locations have positive raw logits. A hard ReLU gate can therefore
        collapse to an all-zero mask. We convert the per-position maximum logit
        to a soft confidence with sigmoid, while still preserving the teacher's
        relative foreground reliability.
        """
        #如何生成用于语义对齐的前景掩码（fg_masks）。
        #SGDALoss 会使用 fg_masks 将学生的注意力集中在“前景”（即有物体的区域），避免让学生去模仿老师在背景区域的无用噪声。
        #这段代码中的 _build_fg_masks_for_sgda 方法，就是利用教师模型自身的“分类预测结果”动态生成这些前景掩码的工程实现。
        fg_masks = []
        text_masks = getattr(self, 'text_masks', None)
#该函数的作用是将教师模型的分类输出（Logits），转化为一张形状为 [Batch, 1, H, W] 的空间权重图。数学表达为 $m_l(p) = [\max_j \text{logit}_t^l(p, j)]_+$。
        for level_idx, tea_cls in enumerate(tea_cls_scores):
            target_h, target_w = tea_reg_hold[level_idx].shape[-2:]
            tea_cls = tea_cls.detach()
#代码检查了 text_masks。如果存在语言掩码（表示某些 padding 的文本 token 是无效的），就会用 masked_fill 将这些无效 token 的预测得分设为 -inf（负无穷大），确保后续取最大值时不会选中无效的文本类别。
            if tea_cls.dim() == 3:
                batch_size, num_positions, num_tokens = tea_cls.shape
                if text_masks is not None and text_masks.dim() == 2:
                    valid_tokens = text_masks[:, None, :num_tokens].to(
                        device=tea_cls.device, dtype=torch.bool)
                    tea_cls = tea_cls.masked_fill(~valid_tokens, float('-inf'))
#B. 提取空间位置的“前景置信度”    掩码的核心逻辑是：“如果教师模型认为某个位置大概率存在某个物体，那么这个位置的权重就应该很高。”
                spatial_positions = target_h * target_w
                if num_positions == spatial_positions:
                    max_logit = tea_cls.max(dim=-1).values
                    max_logit = torch.where(
                        torch.isfinite(max_logit), max_logit,
                        torch.zeros_like(max_logit))
                    mask = max_logit.view(batch_size, 1, target_h, target_w)
                elif num_positions % spatial_positions == 0:
                    num_priors = num_positions // spatial_positions
                    tea_cls = tea_cls.view(
                        batch_size, target_h, target_w, num_priors, num_tokens)
                    max_logit = tea_cls.amax(dim=(-1, -2))
                    max_logit = torch.where(
                        torch.isfinite(max_logit), max_logit,
                        torch.zeros_like(max_logit))
                    mask = max_logit.unsqueeze(1)
                else:
                    max_logit = tea_cls.max(dim=-1).values
                    max_logit = torch.where(
                        torch.isfinite(max_logit), max_logit,
                        torch.zeros_like(max_logit))
                    side = int(num_positions ** 0.5)
                    if side * side == num_positions:
                        mask = max_logit.view(batch_size, 1, side, side)
                    else:
                        mask = max_logit.view(batch_size, 1, 1, num_positions)
                    mask = torch.nn.functional.interpolate(
                        mask, size=(target_h, target_w), mode='nearest')
            elif tea_cls.dim() == 4:
                mask = tea_cls.max(dim=1, keepdim=True).values
                if mask.shape[-2:] != (target_h, target_w):
                    mask = torch.nn.functional.interpolate(
                        mask, size=(target_h, target_w), mode='nearest')
            else:
                raise ValueError(
                    f'Unsupported teacher cls shape for SGDA at level '
                    f'{level_idx}: {tuple(tea_cls.shape)}')
# 使用 sigmoid 将 GLIP token logit 转成软置信度，避免 raw logit 整体为负时 foreground gate 全零。
#修改成了改为 sigmoid soft confidence：
            fg_masks.append(mask.sigmoid())

        return fg_masks

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

        pacs_t2_weights = None
        if self.pacs_cfg.get('enable', False) and self.pacs_router is not None:
            pacs_t2_weights = []
            for feat in stu_x_pseudo:
                w2_eff, w2_raw, phi_sigmoid, eta = self._compute_pacs_t2_weight(
                    feat)
                pacs_t2_weights.append(dict(
                    eff=w2_eff,
                    raw=w2_raw,
                    phi=phi_sigmoid,
                    eta=eta,
                ))

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
            pacs_t2_weights=pacs_t2_weights,
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
        pacs_t2_weights=None,
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

        (losses_cls_align, losses_cls_ent, losses_bbox, losses_centerness,
         stat_off_modulator_mean, stat_off_modulator_active_ratio,
         stat_off_pt_minus_s_mean, stat_pacs_w2_pos_mean, stat_pacs_w2_pos_std,
         stat_pacs_phi_sigmoid, stat_pacs_warmup_eta, losses_pacs_lb) = multi_apply(
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
            pacs_t2_weights if pacs_t2_weights is not None else [None] * len(cls_scores),
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

        loss_cls_align = self._scale_loss(losses_cls_align, pseudo_weight)
        if self.off_bc_cfg.get('scale_entropy_by_scheduler', False):
            loss_cls_ent = self._scale_loss(losses_cls_ent, pseudo_weight)
        else:
            loss_cls_ent = losses_cls_ent

        stat_modulator_mean = torch.stack(
            [item.detach() for item in stat_off_modulator_mean]).mean()
        stat_active_ratio = torch.stack(
            [item.detach() for item in stat_off_modulator_active_ratio]).mean()
        stat_pt_minus_s = torch.stack(
            [item.detach() for item in stat_off_pt_minus_s_mean]).mean()
        stat_pacs_w2_mean = torch.stack(
            [item.detach() for item in stat_pacs_w2_pos_mean]).mean()
        stat_pacs_w2_std = torch.stack(
            [item.detach() for item in stat_pacs_w2_pos_std]).mean()
        stat_pacs_phi = torch.stack(
            [item.detach() for item in stat_pacs_phi_sigmoid]).mean()
        stat_pacs_eta = torch.stack(
            [item.detach() for item in stat_pacs_warmup_eta]).mean()
        loss_pacs_lb = torch.stack(losses_pacs_lb).mean()
        loss_pacs_lb = loss_pacs_lb * stat_pacs_eta * self.pacs_cfg.get(
            'lb_loss_weight', 0.01)

        if self.off_bc_cfg.get('enable', False):
            self._off_bc_iter += 1
            if self._off_bc_iter % 100 == 0:
                align_scalar = self._sum_loss(loss_cls_align).detach().item()
                ent_scalar = self._sum_loss(loss_cls_ent).detach().item()
                print(
                    'OFF-BC sanity: '
                    f'active_ratio={stat_active_ratio.item():.3f}, '
                    f'pt-s={stat_pt_minus_s.item():.3f}, '
                    f'align/ent={align_scalar:.4f}/{ent_scalar:.4f}')

        return dict(
            loss_cls_align=loss_cls_align,
            loss_cls_ent=loss_cls_ent,
            loss_bbox=self._scale_loss(losses_bbox, pseudo_weight),
            loss_centerness=self._scale_loss(losses_centerness, pseudo_weight),
            pseudo_pos_score_mean=pseudo_pos_score_mean,
            stat_off_modulator_mean=stat_modulator_mean,
            stat_off_modulator_active_ratio=stat_active_ratio,
            stat_off_pt_minus_s_mean=stat_pt_minus_s,
            loss_pacs_lb=loss_pacs_lb,
            stat_pacs_w2_mean=stat_pacs_w2_mean,
            stat_pacs_w2_std=stat_pacs_w2_std,
            stat_pacs_phi_sigmoid=stat_pacs_phi,
            stat_pacs_warmup_eta=stat_pacs_eta,
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
        pacs_weight_info=None,
        avg_factor: float = 1.0,
    ):
        """ATSS 单层 anchor-level pseudo-score 加权损失。"""
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.bbox_head.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        pseudo_label_weights = pseudo_label_weights.reshape(-1)
        pseudo_bbox_weights = pseudo_bbox_weights.reshape(-1)

        pacs_w_eff = pacs_w_raw = pacs_phi = pacs_eta = None
        if pacs_weight_info is not None:
            num_imgs = cls_score.shape[0]
            num_anchors = cls_score.shape[2] * cls_score.shape[3]
            pacs_w_eff = pacs_weight_info['eff'].reshape(
                num_imgs, num_anchors).reshape(-1).to(cls_score.device)
            pacs_w_raw = pacs_weight_info['raw'].reshape(
                num_imgs, num_anchors).reshape(-1).to(cls_score.device)
            pacs_phi = pacs_weight_info['phi'].to(cls_score.device)
            pacs_eta = pacs_weight_info['eta'].to(cls_score.device)

        bg_class_ind = self.bbox_head.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        off_bc_enabled = self.off_bc_cfg.get('enable', False)
        zero_stat = cls_score.new_tensor(0.0)

        if off_bc_enabled and len(pos_inds) > 0:
            (loss_cls_align, loss_cls_ent, stat_off_modulator_mean,
             stat_off_modulator_active_ratio, stat_off_pt_minus_s_mean) = \
                self._compute_off_bc_cls_loss(
                    cls_score=cls_score,
                    labels=labels,
                    label_weights=label_weights,
                    pos_inds=pos_inds,
                    pseudo_label_weights=pseudo_label_weights,
                    pacs_weights=pacs_w_eff,
                    avg_factor=avg_factor,
                )
        else:
            final_label_weights = label_weights.clone()
            if len(pos_inds) > 0:
                final_label_weights[pos_inds] *= pseudo_label_weights[pos_inds]
                if pacs_w_eff is not None:
                    final_label_weights[pos_inds] *= pacs_w_eff[pos_inds]
            loss_cls_align = self.bbox_head.loss_cls(
                cls_score, labels, final_label_weights, avg_factor=avg_factor)
            loss_cls_ent = cls_score.sum() * 0
            stat_off_modulator_mean = zero_stat
            stat_off_modulator_active_ratio = zero_stat
            stat_off_pt_minus_s_mean = zero_stat

        if len(pos_inds) > 0:
            if off_bc_enabled:
                reg_floor = self.off_bc_cfg.get('reg_score_floor', 0.5)
                reg_mask = pseudo_bbox_weights[pos_inds] >= reg_floor
                reg_pos_inds = pos_inds[reg_mask]
            else:
                reg_pos_inds = pos_inds

            if len(reg_pos_inds) > 0:
                pos_bbox_targets = bbox_targets[reg_pos_inds]
                pos_bbox_pred = bbox_pred[reg_pos_inds]
                pos_anchors = anchors[reg_pos_inds]
                pos_centerness = centerness[reg_pos_inds]
                pos_pseudo_bbox_weights = pseudo_bbox_weights[reg_pos_inds]

                if pacs_w_eff is not None:
                    effective_reg_weight = (
                        pos_pseudo_bbox_weights * pacs_w_eff[reg_pos_inds])
                else:
                    effective_reg_weight = pos_pseudo_bbox_weights

                centerness_targets = self.bbox_head.centerness_target(
                    pos_anchors, pos_bbox_targets)
                pos_decode_bbox_pred = self.bbox_head.bbox_coder.decode(
                    pos_anchors, pos_bbox_pred)

                loss_bbox = self.bbox_head.loss_bbox(
                    pos_decode_bbox_pred,
                    pos_bbox_targets,
                    weight=centerness_targets * effective_reg_weight,
                    avg_factor=avg_factor)

                loss_centerness = self.bbox_head.loss_centerness(
                    pos_centerness,
                    centerness_targets,
                    weight=effective_reg_weight,
                    avg_factor=avg_factor)
            else:
                loss_bbox = cls_score.sum() * 0
                loss_centerness = cls_score.sum() * 0
        else:
            loss_bbox = cls_score.sum() * 0
            loss_centerness = cls_score.sum() * 0

        if pacs_w_raw is not None and len(pos_inds) > 0:
            pos_raw = pacs_w_raw[pos_inds]
            stat_pacs_w2_pos_mean = pos_raw.mean()
            stat_pacs_w2_pos_std = pos_raw.std(unbiased=False)
            stat_pacs_phi_sigmoid = pacs_phi
            stat_pacs_warmup_eta = pacs_eta
            loss_pacs_lb = self.pacs_router.compute_lb_loss(
                stat_pacs_w2_pos_mean, pacs_phi)
        else:
            stat_pacs_w2_pos_mean = zero_stat
            stat_pacs_w2_pos_std = zero_stat
            stat_pacs_phi_sigmoid = zero_stat
            stat_pacs_warmup_eta = zero_stat
            loss_pacs_lb = zero_stat

        return (loss_cls_align, loss_cls_ent, loss_bbox, loss_centerness,
                stat_off_modulator_mean, stat_off_modulator_active_ratio,
                stat_off_pt_minus_s_mean, stat_pacs_w2_pos_mean,
                stat_pacs_w2_pos_std, stat_pacs_phi_sigmoid,
                stat_pacs_warmup_eta, loss_pacs_lb)

    def _compute_off_bc_cls_loss(
        self,
        cls_score: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        pos_inds: Tensor,
        pseudo_label_weights: Tensor,
        avg_factor: float,
        pacs_weights: Optional[Tensor] = None,
    ):
        """OFF-BC classification loss for pseudo-positive anchors.

        L = [max(0, s - p_t)]^gamma * BCE(p_t, s)
          + lambda * (1 - s)^delta * KL_bin(y_hat || Bernoulli(0.5)).
        """
        cfg = self.off_bc_cfg
        gamma = cfg.get('gamma', 2.0)
        delta = cfg.get('delta', 2.0)
        lambda_ent = cfg.get('lambda_ent', 0.05)
        detach_modulator = cfg.get('detach_modulator', False)

        pos_cls_score = cls_score[pos_inds]
        pos_labels = labels[pos_inds]
        pos_s = pseudo_label_weights[pos_inds].clamp(0, 1)
        pos_pacs_w = pacs_weights[pos_inds] if pacs_weights is not None else None

        pos_logit_y = pos_cls_score.gather(
            1, pos_labels.unsqueeze(1)).squeeze(1)
        pos_p_t = pos_logit_y.sigmoid()

        gap = (pos_s - pos_p_t).clamp(min=0.0)
        if detach_modulator:
            gap = gap.detach()
        modulator = gap.pow(gamma)

        bce = F.binary_cross_entropy_with_logits(
            pos_logit_y, pos_s, reduction='none')
        if pos_pacs_w is not None:
            loss_align = (pos_pacs_w * modulator * bce).sum() / \
                modulator.sum().clamp(min=1.0)
        else:
            loss_align = (modulator * bce).sum() / modulator.sum().clamp(min=1.0)

        pseudo_neg_weight = cfg.get('pseudo_neg_weight', 0.0)
        if pseudo_neg_weight > 0:
            bg_class_ind = self.bbox_head.num_classes
            neg_inds = (labels == bg_class_ind).nonzero().squeeze(1)
            if len(neg_inds) > 0:
                neg_label_weights = label_weights.new_zeros(label_weights.shape)
                neg_label_weights[neg_inds] = label_weights[neg_inds] * pseudo_neg_weight
                loss_align = loss_align + self.bbox_head.loss_cls(
                    cls_score, labels, neg_label_weights, avg_factor=avg_factor)

        prob = pos_cls_score.sigmoid().clamp(1e-6, 1 - 1e-6)
        binary_entropy = -(
            prob * prob.log() + (1 - prob) * (1 - prob).log())
        kl_to_uniform = (math.log(2.0) - binary_entropy).mean(dim=-1)
        ent_weight = (1.0 - pos_s).pow(delta)
        if pos_pacs_w is not None:
            loss_ent = lambda_ent * (pos_pacs_w * ent_weight * kl_to_uniform).sum() / \
                ent_weight.sum().clamp(min=1.0)
        else:
            loss_ent = lambda_ent * (
                ent_weight * kl_to_uniform).sum() / ent_weight.sum().clamp(min=1.0)

        with torch.no_grad():
            stat_off_modulator_mean = modulator.mean()
            stat_off_modulator_active_ratio = (
                modulator > 1e-6).float().mean()
            stat_off_pt_minus_s_mean = (pos_p_t - pos_s).mean()

        return (loss_align, loss_ent, stat_off_modulator_mean,
                stat_off_modulator_active_ratio, stat_off_pt_minus_s_mean)

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

    @staticmethod
    def _sum_loss(loss_value):
        if isinstance(loss_value, list):
            return sum(loss_value)
        return loss_value


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


