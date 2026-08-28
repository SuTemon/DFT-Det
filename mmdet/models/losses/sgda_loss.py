# Copyright (c) FusionKD. All rights reserved.
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmdet.registry import MODELS


@MODELS.register_module()
class SGDALoss(BaseModule):
    """Semantic-Geometric Dual Alignment loss for cross-modal KD.

    The loss decouples intermediate feature alignment into foreground-gated
    semantic direction alignment and position-wide geometric structure
    alignment.
    """

    def __init__(
        self,
        student_channels: int = 256,
        teacher_channels: int = 256,
        num_levels: int = 5,
        downsample_ratios: Sequence[int] = (4, 2, 1, 1, 1),
        loss_sem_weight: float = 1.0,
        loss_geo_weight: float = 1.0,
        fg_eps: float = 1e-6,
        gram_l2_normalize: bool = True,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        if len(downsample_ratios) != num_levels:
            raise ValueError(
                'downsample_ratios length must equal num_levels, got '
                f'{len(downsample_ratios)} and {num_levels}')

        self.num_levels = num_levels
        self.downsample_ratios = tuple(downsample_ratios)
        self.loss_sem_weight = loss_sem_weight
        self.loss_geo_weight = loss_geo_weight
        self.fg_eps = fg_eps
        self.gram_l2_normalize = gram_l2_normalize

        if student_channels == teacher_channels:
            self.psi_s = nn.Identity()
        else:
            self.psi_s = nn.Conv2d(
                student_channels, teacher_channels, kernel_size=1, bias=False)
            nn.init.kaiming_normal_(self.psi_s.weight, mode='fan_out')
#语义对齐损失，作用：只关注图像中的关键区域（前景），让学生模型学到与教师模型一致的语义特征（特征向量的方向）。
    def _semantic_loss_single_level(self, feat_s: torch.Tensor,
                                    feat_t: torch.Tensor,
                                    fg_mask: torch.Tensor) -> torch.Tensor:
        cos_sim = F.cosine_similarity(feat_s, feat_t, dim=1)
        cos_dist = 1.0 - cos_sim
#前景门控权重，只关注前景区域，背景区域权重为0。
        fg_weight = fg_mask.squeeze(1).to(dtype=cos_dist.dtype)
        weighted = cos_dist * fg_weight
        numerator = weighted.sum(dim=(1, 2))
        denominator = fg_weight.sum(dim=(1, 2)).clamp_min(self.fg_eps)
        return (numerator / denominator).mean()
#几何结构对齐损失，作用：关注全局的结构和位置关系，让学生模型理解物体之间的空间几何拓扑。
    def _geometric_loss_single_level(self, feat_s: torch.Tensor,
                                     feat_t: torch.Tensor,
                                     downsample_ratio: int) -> torch.Tensor:
        _, _, h, w = feat_s.shape
        #下采样，减小计算量，提高效率。
        if downsample_ratio > 1:
            target_h = max(h // downsample_ratio, 1)
            target_w = max(w // downsample_ratio, 1)
            feat_s = F.adaptive_avg_pool2d(feat_s, (target_h, target_w))
            feat_t = F.adaptive_avg_pool2d(feat_t, (target_h, target_w))

        feat_s_flat = feat_s.flatten(2).transpose(1, 2)
        feat_t_flat = feat_t.flatten(2).transpose(1, 2)

        if self.gram_l2_normalize:
            feat_s_flat = F.normalize(feat_s_flat, p=2, dim=2)
            feat_t_flat = F.normalize(feat_t_flat, p=2, dim=2)
#计算特征的Gram矩阵，捕捉特征之间的全局相关性。通过 torch.bmm 计算特征在空间位置上的自相关矩阵（Gram Matrix），形状为 [Batch, 像素点数, 像素点数]。
        gram_s = torch.bmm(feat_s_flat, feat_s_flat.transpose(1, 2))
        gram_t = torch.bmm(feat_t_flat, feat_t_flat.transpose(1, 2))
        num_positions = feat_s_flat.size(1)
#结构对齐：对学生和教师的 Gram Matrix 计算均方误差（MSE）。这意味着它不强求学生在某个像素点上的特征和教师一模一样，而是要求学生特征图的空间内部关系网（哪部分和哪部分相似）与教师保持一致。
        loss = (gram_s - gram_t).pow(2).sum(dim=(1, 2))
        loss = loss / max(num_positions * num_positions, 1)
        return loss.mean()
#接收多层特征金字塔的特征 (feats_s, feats_t) 以及每层对应的前景掩码 (fg_masks)。
    def forward(self, feats_s, feats_t, fg_masks):
        if len(feats_s) != self.num_levels or len(feats_t) != self.num_levels:
            raise ValueError(
                f'SGDALoss expects {self.num_levels} levels, got '
                f'{len(feats_s)} student and {len(feats_t)} teacher levels')
        if len(fg_masks) != self.num_levels:
            raise ValueError(
                f'SGDALoss expects {self.num_levels} fg masks, '
                f'got {len(fg_masks)}')

        total_sem = feats_s[0].new_zeros(())
        total_geo = feats_s[0].new_zeros(())

        for level_idx in range(self.num_levels):
            feat_s = feats_s[level_idx]
            feat_t = feats_t[level_idx].detach()
            fg_mask = fg_masks[level_idx].detach()

            feat_s = self.psi_s(feat_s)
            if feat_s.shape != feat_t.shape:
                raise ValueError(
                    f'SGDA feature shape mismatch at level {level_idx}: '
                    f'student {tuple(feat_s.shape)} vs teacher '
                    f'{tuple(feat_t.shape)}')
            if fg_mask.shape[-2:] != feat_t.shape[-2:]:
                fg_mask = F.interpolate(
                    fg_mask, size=feat_t.shape[-2:], mode='nearest')

            total_sem = total_sem + self._semantic_loss_single_level(
                feat_s, feat_t, fg_mask)
            total_geo = total_geo + self._geometric_loss_single_level(
                feat_s, feat_t, self.downsample_ratios[level_idx])
#最终返回一个字典，包含乘以对应权重（loss_sem_weight 和 loss_geo_weight）的两种损失，供优化器进行反向传播。
        return {
            'loss_sgda_sem': total_sem * self.loss_sem_weight,
            'loss_sgda_geo': total_geo * self.loss_geo_weight,
        }
#语义层面上，抛弃了容易受模态差异影响的绝对值约束，改为带前景掩码的方向（余弦）约束。
#几何层面上，不进行像素对像素的死板对齐，而是对齐特征图的内部空间拓扑关系（Gram矩阵）。