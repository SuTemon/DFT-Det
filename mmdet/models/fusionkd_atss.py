# Copyright (c) OpenMMLab. All rights reserved.
# 逐行关键注释说明：本文件实现 FusionKD-ATSS 检测器，核心流程是学生特征、教师视觉语言特征、跨模态融合特征与复用教师头预测之间的蒸馏。
# 注释重点标出每个关键变量在检测损失、分类/回归/中心度 KD、文本增强和教师头复用中的作用。
from typing import Dict, List, Union, Any

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


from mmdet.registry import MODELS  # 导入依赖模块/类型，供模型构建、张量计算和可视化使用。
from mmdet.utils import (InstanceList, OptInstanceList, OptConfigType, reduce_mean)  # 导入依赖模块/类型，供模型构建、张量计算和可视化使用。
from .Fusionkd_single_stage import FusionKDSingleStageDetector  # 导入依赖模块/类型，供模型构建、张量计算和可视化使用。

from ..utils import permute_and_flatten  # 导入依赖模块/类型，供模型构建、张量计算和可视化使用。
from ..utils import images_to_levels, multi_apply, unpack_gt_instances  # 导入依赖模块/类型，供模型构建、张量计算和可视化使用。
import re  # 导入依赖模块/类型，供模型构建、张量计算和可视化使用。
from typing import Optional, Tuple, Union  # 导入依赖模块/类型，供模型构建、张量计算和可视化使用。
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig  # 导入依赖模块/类型，供模型构建、张量计算和可视化使用。
from mmdet.structures import SampleList  # 导入依赖模块/类型，供模型构建、张量计算和可视化使用。
from mmdet.structures.bbox import cat_boxes  # 导入依赖模块/类型，供模型构建、张量计算和可视化使用。

import numpy as np  # 导入依赖模块/类型，供模型构建、张量计算和可视化使用。
import matplotlib.pyplot as plt  # 导入依赖模块/类型，供模型构建、张量计算和可视化使用。
from sklearn.decomposition import PCA  # 导入依赖模块/类型，供模型构建、张量计算和可视化使用。
from skimage.transform import resize  # 导入依赖模块/类型，供模型构建、张量计算和可视化使用。
import torch  # 导入依赖模块/类型，供模型构建、张量计算和可视化使用。

class DefectAwarePromptConstructor:
    """Defect-Aware Prompt Construction (DAPC).

    For each defect class c_k, M prompt templates are constructed from three
    complementary dimensions:
      (i)  Texture  — surface appearance / material pattern
      (ii) Morphology — geometric shape / spatial extent
      (iii) Contrast — discriminative boundary against background

    The robust text prototype is the mean of all M text embeddings::

        ē_k = (1/M) Σ_{m=1}^{M} f_t(p_k^(m))   ... Eq.(8)

    In GLIP/FusionKD the classification head is token-level, so this
    implementation ensembles only the token span corresponding to ``{cls}``
    in each prompt, not the whole sentence.  At runtime the resulting class
    token prototypes are injected back into the real caption token positions
    given by ``tokens_positive``.
    """

    DEFAULT_DESCRIPTIONS: Dict[str, str] = {
        'crazing': (
            'dense network of fine hairline cracks on hot-rolled steel surface'),
        'inclusion': (
            'embedded non-metallic foreign material appearing as dark elongated streaks'),
        'patches': (
            'irregular patch-like oxide stains with darker tone than surrounding steel'),
        'pitted surface': (
            'cluster of small dark circular pits distributed over the metal surface'),
        'rolled-in scale': (
            'dark flaky oxide scale pressed into the steel during hot rolling'),
        'scratches': (
            'long thin linear scratches with straight trajectory on steel sheet'),
    }

    TEMPLATES: List[str] = [
        'a close-up grayscale image of hot-rolled steel showing {cls}, which is {desc}',
        'a metal surface defect named {cls}: {desc}',
        'industrial inspection image with {cls} visible on steel surface, {desc}',
        'distinguishing {cls} from normal background texture, described as {desc}',
        'surface texture pattern of {cls} defect on rolled steel sheet, {desc}',
    ]

    def __init__(
        self,
        class_names: List[str],
        descriptions: Optional[Dict[str, str]] = None,
        templates: Optional[List[str]] = None,
    ):
        self.class_names = [self.normalise_class_name(name) for name in class_names]
        self.descriptions = descriptions or self.DEFAULT_DESCRIPTIONS
        self.templates = templates or self.TEMPLATES
        self.num_classes = len(self.class_names)
        self.M = len(self.templates)

    @staticmethod
    def normalise_class_name(name: str) -> str:
        name = re.sub(r'\(.*\)', '', name)
        name = re.sub(r'_', ' ', name)
        name = re.sub(r'\s+', ' ', name)
        return name.strip()

    @staticmethod
    def _char_to_token(tokenized, char_idx: int):
        try:
            return tokenized.char_to_token(char_idx)
        except TypeError:
            return tokenized.char_to_token(0, char_idx)

    def get_prompts(self, class_idx: int) -> List[str]:
        cls = self.class_names[class_idx]
        desc = self.descriptions.get(cls, f'{cls} defect on steel surface')
        return [t.format(cls=cls, desc=desc) for t in self.templates]

    def _get_class_token_span(self, tokenized, prompt: str, class_name: str):
        cls_start = prompt.lower().find(class_name.lower())
        if cls_start < 0:
            return []
        cls_end = cls_start + len(class_name)
        beg_pos = self._char_to_token(tokenized, cls_start)
        end_pos = self._char_to_token(tokenized, cls_end - 1)
        if beg_pos is None:
            for offset in (1, 2):
                beg_pos = self._char_to_token(tokenized, cls_start + offset)
                if beg_pos is not None:
                    break
        if end_pos is None:
            for offset in (2, 3):
                end_pos = self._char_to_token(tokenized, cls_end - offset)
                if end_pos is not None:
                    break
        if beg_pos is None or end_pos is None or end_pos < beg_pos:
            return []
        return list(range(beg_pos, end_pos + 1))

    @torch.no_grad()
    def build_robust_token_prototypes(self, language_model):
        """Compute token-level robust prototypes for all classes.

        Returns:
            tuple[Tensor, Tensor]:
              - prototypes: ``(K, L_cls_max, D)`` padded class-token prototypes
              - lengths: ``(K,)`` valid token count for each class name
        """
        class_prototypes = []
        class_lengths = []
        max_len = 1

        for k, class_name in enumerate(self.class_names):
            prompts = self.get_prompts(k)
            prompt_spans = []
            for prompt in prompts:
                out = language_model([prompt])
                tokenized = language_model.tokenizer([prompt], return_tensors='pt')
                token_indices = self._get_class_token_span(tokenized, prompt, class_name)
                if not token_indices:
                    continue
                emb = out['embedded'][0, token_indices, :]
                prompt_spans.append(emb)

            if not prompt_spans:
                raise RuntimeError(
                    f'DAPC failed to locate class tokens for class "{class_name}".')

            cls_len = max(span.shape[0] for span in prompt_spans)
            max_len = max(max_len, cls_len)
            proto_sum = prompt_spans[0].new_zeros(cls_len, prompt_spans[0].shape[-1])
            proto_count = prompt_spans[0].new_zeros(cls_len, 1)
            for span in prompt_spans:
                cur_len = span.shape[0]
                proto_sum[:cur_len] += span
                proto_count[:cur_len] += 1
            proto = proto_sum / proto_count.clamp_min(1.0)
            proto = F.normalize(proto, p=2, dim=-1)
            class_prototypes.append(proto)
            class_lengths.append(cls_len)

        dim = class_prototypes[0].shape[-1]
        padded = class_prototypes[0].new_zeros(self.num_classes, max_len, dim)
        for k, proto in enumerate(class_prototypes):
            padded[k, :proto.shape[0], :] = proto
        lengths = torch.tensor(class_lengths, dtype=torch.long, device=padded.device)
        return padded, lengths


def visualize_channel_avg(x):  # 定义 visualize_channel_avg 函数/方法，承担当前蒸馏流程中的一个子步骤。
    """
    可视化特征金字塔中各层的平均通道特征

    参数:
    x (tuple): 包含多个特征层的元组, 每个层形状为 [1, 256, H, W]
    """
    # 检查输入是否为元组
    # 使用方法:
    # x = self.extract_feat(batch_inputs)  # 获取特征元组
    # visualize_channel_avg(x)
    import torch  # 导入依赖模块/类型，供模型构建、张量计算和可视化使用。
    import torch.nn.functional as F  # 导入依赖模块/类型，供模型构建、张量计算和可视化使用。
    import matplotlib.pyplot as plt  # 导入依赖模块/类型，供模型构建、张量计算和可视化使用。

    #
    # if not isinstance(x, tuple):
    #     print("Error: Input should be a tuple of feature maps")
    #     return

    num_levels = len(x)

    # 创建子图
    fig, axes = plt.subplots(1, num_levels, figsize=(18, 4))
    if num_levels == 1:  # 条件分支，根据当前状态选择不同处理路径。
        axes = [axes]  # 确保单层时axes仍是列表

    # 确定最大原始尺寸用于归一化
    max_size = max([feat.shape[2] for feat in x])

    for i, feat in enumerate(x):  # 循环处理 batch、FPN 层或类别维度，保持多尺度逻辑一致。
        # 计算通道平均值 [1, H, W]
        channel_avg = torch.mean(feat.detach(), dim=1)[0]  # [0] 移除批次维度 => [H, W]

        # 归一化到 [0, 1]
        min_val = torch.min(channel_avg)
        max_val = torch.max(channel_avg)
        normalized = (channel_avg - min_val) / (max_val - min_val + 1e-8)

        # 上采样到统一尺寸便于比较
        upsampled = F.interpolate(
            normalized[None, None, ...],  # 添加批次和通道维度
            size=(max_size, max_size),
            mode='bilinear',
            align_corners=False
        )[0, 0]  # 移除添加的维度


        # 可视化
        ax = axes[i]
        im = ax.imshow(upsampled.cpu().numpy(), cmap='viridis')
        fig.colorbar(im, ax=ax, shrink=0.7)  # 添加颜色条

        # 添加原始分辨率信息
        orig_shape = feat.shape
        title = f'Level {i}\n({orig_shape[2]}×{orig_shape[3]})'
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    plt.suptitle('Average Channel Features Across Levels', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()


@MODELS.register_module()  # 装饰器注册/修饰后续类或方法，保证框架能发现该模块。
class FusionKDATSS(FusionKDSingleStageDetector):  # 定义 FusionKDATSS 类，封装检测器或跨模态辅助模块。

    def __init__(self,  # 定义 __init__ 函数/方法，承担当前蒸馏流程中的一个子步骤。
                 kd_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super().__init__(kd_cfg=kd_cfg,**kwargs)  # 调用父类初始化，继承单阶段检测器和教师学生蒸馏基础能力。
        # 添加动态蒸馏权重控制
        self.kd_weight_scheduler = {  # 初始化/更新 self.kd_weight_scheduler，作为模型状态或蒸馏模块配置。
            'cls': 1.0, 'reg': 1.0, 'center': 1.0, 'feat': 0.5
        }
        self.current_epoch = 0  # 初始化/更新 self.current_epoch，作为模型状态或蒸馏模块配置。
        self.total_epochs = kwargs.get('train_cfg', {}).get('max_epochs', 30)  # 初始化/更新 self.total_epochs，作为模型状态或蒸馏模块配置。

        # 梯度协调参数
        self.gradient_stats = {  # 初始化/更新 self.gradient_stats，作为模型状态或蒸馏模块配置。
            'det_loss': 0.0,
            'kd_loss': 0.0,
            'ratio_threshold': 0.6  # KD损失不应超过总损失的60%
        }

        # 性能监控
        self.best_map = 0.0  # 初始化/更新 self.best_map，作为模型状态或蒸馏模块配置。
        self.performance_history = []  # 初始化/更新 self.performance_history，作为模型状态或蒸馏模块配置。

        self.loss_center_kd = None  # 初始化/更新 self.loss_center_kd，作为模型状态或蒸馏模块配置。
        if kd_cfg.get('loss_center_kd', None):  # 条件分支，根据当前状态选择不同处理路径。
            self.loss_center_kd = MODELS.build(kd_cfg['loss_center_kd'])  # 初始化/更新 self.loss_center_kd，作为模型状态或蒸馏模块配置。
        #add by jojo
        # 文本适配模块
        self.text_dim=768  # 初始化/更新 self.text_dim，作为模型状态或蒸馏模块配置。
        self.text_proj = nn.Sequential(  # 初始化/更新 self.text_proj，作为模型状态或蒸馏模块配置。
            nn.Linear(self.text_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True)
        )
        #
        # 跨模态注意力
        self.cross_attn = CrossModalAttention(  # 初始化/更新 self.cross_attn，作为模型状态或蒸馏模块配置。
            v_dim=256,  # 学生特征维度
            l_dim=256,  # 投影后文本维度 512-256
            num_heads=8)

        # 文本增强的FPN
        self.text_aware_fpn = nn.ModuleList([  # 初始化/更新 self.text_aware_fpn，作为模型状态或蒸馏模块配置。
            TextAwareFPNLayer(256)
            for _ in range(5)  # 循环处理 batch、FPN 层或类别维度，保持多尺度逻辑一致。
        ])

        #visual enhanced text
        self.text_enhancer = VisualEnhancedText(text_dim=768)  # 初始化/更新 self.text_enhancer，作为模型状态或蒸馏模块配置。
        #add end

        # ------------------------------------------------------------------
        # Defect-Aware Prompt Construction (DAPC) — Sec. 3.1.2
        # DAPC is computed once because the current distillation teacher is
        # frozen by default. If the language model is unfrozen in an ablation,
        # call refresh_dapc_prototypes() after updating the teacher.
        # ------------------------------------------------------------------
        dapc_cfg = {} if kd_cfg is None else kd_cfg.get('dapc_cfg', {})
        self.enable_dapc = dapc_cfg.get('enable', True)
        self.dapc_alpha = dapc_cfg.get('alpha', 0.35)
        dapc_class_names = dapc_cfg.get(
            'class_names',
            ['crazing', 'inclusion', 'patches',
             'pitted_surface', 'rolled-in_scale', 'scratches'])
        self._dapc = DefectAwarePromptConstructor(
            dapc_class_names,
            descriptions=dapc_cfg.get('descriptions', None),
            templates=dapc_cfg.get('templates', None))
        self.refresh_dapc_prototypes()

    def refresh_dapc_prototypes(self) -> None:
        """Recompute DAPC prototypes from the current teacher language model.

        The default setting freezes the teacher, so this is called once during
        initialization. If an ablation unfreezes ``teacher.language_model``,
        call this method after teacher updates to keep prototypes consistent.
        """
        if not getattr(self, 'enable_dapc', False):
            return
        with torch.no_grad():
            prototypes, lengths = self._dapc.build_robust_token_prototypes(
                self.teacher.language_model)
        if 'dapc_prototypes' in self._buffers:
            self.dapc_prototypes = prototypes
            self.dapc_token_lengths = lengths
        else:
            self.register_buffer('dapc_prototypes', prototypes, persistent=False)
            self.register_buffer('dapc_token_lengths', lengths, persistent=False)

    def update_kd_weights(self, epoch, strategy='adaptive'):  # 定义 update_kd_weights 函数/方法，承担当前蒸馏流程中的一个子步骤。
        """根据训练进度动态调整蒸馏权重"""
        self.current_epoch = epoch  # 初始化/更新 self.current_epoch，作为模型状态或蒸馏模块配置。
        progress = epoch / self.total_epochs

        if strategy == 'adaptive':  # 条件分支，根据当前状态选择不同处理路径。
            # 自适应衰减策略
            if epoch < 10:  # 条件分支，根据当前状态选择不同处理路径。
                # 前期：强蒸馏，快速学习教师知识
                decay_factor = 1.0 - progress * 0.2
            elif epoch < 20:  # 追加条件分支，处理另一种训练或数据状态。
                # 中期：平衡蒸馏，防止过拟合
                decay_factor = 0.8 - (progress - 0.33) * 0.4
            else:  # 兜底分支，处理上述条件都不满足的情况。
                # 后期：弱蒸馏，主要优化检测任务
                decay_factor = max(0.2, 0.4 - (progress - 0.66) * 0.5)

        elif strategy == 'cosine':  # 追加条件分支，处理另一种训练或数据状态。
            # 余弦衰减
            import math  # 导入依赖模块/类型，供模型构建、张量计算和可视化使用。
            decay_factor = 0.5 * (1 + math.cos(math.pi * progress))

        elif strategy == 'performance_aware':  # 追加条件分支，处理另一种训练或数据状态。
            # 性能感知衰减
            if len(self.performance_history) > 3:  # 条件分支，根据当前状态选择不同处理路径。
                recent_improvement = self.performance_history[-1] - self.performance_history[-3]
                if recent_improvement < 0.01:  # 性能提升缓慢
                    decay_factor = max(0.2, decay_factor * 0.8)
                else:  # 兜底分支，处理上述条件都不满足的情况。
                    decay_factor = min(1.0, decay_factor * 1.1)

        self.kd_weight_scheduler = {  # 初始化/更新 self.kd_weight_scheduler，作为模型状态或蒸馏模块配置。
            'cls': decay_factor,
            'reg': decay_factor * 0.8,  # 回归任务权重稍低
            'center': decay_factor,
            'feat': 0.3 * decay_factor  # 特征蒸馏权重最低
        }

        return self.kd_weight_scheduler  # 返回当前步骤的结果，供上层训练流程继续使用。

    def check_gradient_balance(self, logger):  # 定义 check_gradient_balance 函数/方法，承担当前蒸馏流程中的一个子步骤。
        """检查梯度平衡，防止蒸馏损失主导"""
        det_loss = self.gradient_stats['det_loss']
        kd_loss = self.gradient_stats['kd_loss']
        total_loss = det_loss + kd_loss + 1e-8

        kd_ratio = kd_loss / total_loss

        if kd_ratio > self.gradient_stats['ratio_threshold']:  # 条件分支，根据当前状态选择不同处理路径。
            # KD损失过大，需要调整
            adjustment = self.gradient_stats['ratio_threshold'] / kd_ratio
            for key in self.kd_weight_scheduler:  # 循环处理 batch、FPN 层或类别维度，保持多尺度逻辑一致。
                self.kd_weight_scheduler[key] *= adjustment  # 初始化/更新 self.kd_weight_scheduler[key] *，作为模型状态或蒸馏模块配置。

            if logger and self.current_epoch % 10 == 0:  # 条件分支，根据当前状态选择不同处理路径。
                logger.info(f'Gradient balance adjusted: KD ratio {kd_ratio:.3f}')

    def loss(self, batch_inputs: Tensor,  # 定义 loss 函数/方法，承担当前蒸馏流程中的一个子步骤。
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        #student branch
        stu_x = self.extract_feat(batch_inputs)  # 学生分支提取 FPN 多尺度视觉特征，后续既用于正常检测也用于蒸馏对齐。
        # visualize_channel_avg(stu_x)
        # for i in range(len(stu_x)):
            # print("stu:",stu_x[i].shape)
            # stu: torch.Size([2, 256, 100, 100])
            # stu: torch.Size([2, 256, 50, 50])
            # stu: torch.Size([2, 256, 25, 25])
            # stu: torch.Size([2, 256, 13, 13])
            # stu: torch.Size([2, 256, 7, 7])
        # stu_cls_scores, stu_bbox_preds, stu_centernesses, stu_cls_hold, stu_reg_hold = \
        #     multi_apply(self.forward_hkd_single,
        #                 stu_x,
        #                 self.bbox_head.scales,
        #                 module=self)
        #---------------------------------------------------


        #teacher branch--------------------------------------
        tea_x = self.teacher.extract_feat(batch_inputs)  # 教师分支提取同尺度视觉特征，作为知识蒸馏的高质量目标。
        # visualize_channel_avg( tea_x)
        # for i in range(len( tea_x )):
        #     print( "tea:",tea_x [i].shape)
        #     tea: torch.Size([2, 256, 100, 100])
        #     tea: torch.Size([2, 256, 50, 50])
        #     tea: torch.Size([2, 256, 25, 25])
        #     tea: torch.Size([2, 256, 13, 13])
        #     tea: torch.Size([2, 256, 7, 7])

        tea_cls_scores, tea_bbox_preds, tea_centernesses, tea_cls_hold, tea_reg_hold = \
            self.forward_crosskd_single_tea(
                        tea_x,
                        batch_inputs,
                        batch_data_samples,
                        module=self.teacher)
        # for i in range(len(tea_cls_scores)):
        #     print("tea_cls_scores:", tea_cls_scores[i].shape)#[1,10000,256]
        #     print("tea_bbox_preds:", tea_bbox_preds[i].shape)#[1,4,100/50/25/13/7,100/50/25/13/7]
        #     print("tea_centernesses:", tea_centernesses[i].shape)#[1,1,100/50/25/13/7,100/50/25/13/7]


        #--------------------------------------------------------
        # 执行可视化
        # 3. 执行可视化
        # 执行可视化
        # 执行可视化
        # visualize_features(
        #     tea_cls_scores,
        #     tea_bbox_preds,
        #     tea_centernesses,
        #     mode='max',  # 可选 'pca' 或 'max'
        #     target_size=300,  # 统一尺寸
        #     save_path='./feature_visualization_tea'  # 保存路径
        # )
        # #----------------------------------------------



        text_embeds=tea_cls_hold## [2,256,768]
        # 文本投影
        proj_text = self.text_proj(text_embeds)# [2,256,512]
        # 文本增强的特征金字塔
        enhanced_feats = []  # 保存文本门控增强后的每层学生 FPN 特征。
        for i, feat in enumerate( stu_x ):  # 循环处理 batch、FPN 层或类别维度，保持多尺度逻辑一致。
            enhanced = self.text_aware_fpn[i](feat, proj_text)  # 使用文本语义生成通道门控，增强当前尺度的学生视觉特征。
            enhanced_feats.append(enhanced)
            # print("enhanced:",enhanced.shape) #same
        # visualize_channel_avg(enhanced_feats)
        # print("visualize_channel_avg(enhanced_feats) done !")

        # 跨模态注意力融合
        fused_feats = [self.cross_attn(f, proj_text) for f in enhanced_feats]  # 通过跨模态注意力进一步让视觉位置关注文本 token，得到融合特征。
        # visualize_channel_avg(fused_feats)
        # print("visualize_channel_avg(fused_feats) done !")


        # for i in range(len(fused_feats)):
        #     print("fused_feats",fused_feats[i].shape)#same
        #     # torch.Size([1, 256, 100, 100])
        #     # torch.Size([1, 256, 50, 50])
        #     # torch.Size([1, 256, 25, 25])
        #     # torch.Size([1, 256, 13, 13])
        #     # torch.Size([1, 256, 7, 7])


        # stu_cls_scores, stu_bbox_preds, stu_centernesses, stu_cls_hold, stu_reg_hold = \
        #     multi_apply(self.forward_hkd_single,
        #                 stu_x,
        #                 self.bbox_head.scales,#ori
        #                 module=self)#result-74.10

        # 分析特征空间


        stu_cls_scores, stu_bbox_preds, stu_centernesses, _, _ = \
            multi_apply(self.forward_hkd_single,
                        stu_x,
                        self.bbox_head.scales,  # ori
                        module=self)  # result-74.10

        # 执行可视化（无边界框）
        # 执行可视化
        # 3. 执行可视化
        # 执行可视化
        # 执行可视化
        # visualize_features(
        #     stu_cls_scores,
        #     stu_bbox_preds,
        #     stu_centernesses,
        #     mode='max',  # 可选 'pca' 或 'max'
        #     target_size=300,  # 统一尺寸
        #     save_path='./feature_visualization_stu'  # 保存路径
        # )

        # for i in range(len(stu_cls_scores)):
        #     print("fused_feats",stu_cls_scores[i].shape) # torch.Size([1, 80, 100, 100])
        #     print("fused_feats",  stu_bbox_preds[i].shape)  # torch.Size([1, 4, 100, 100])
        #     print("stu_centernesses:",stu_centernesses[i].shape) #stu_centernesses: torch.Size([1, 1, 100, 100])

        # for i in range(len(tea_cls_scores)):
        #     print("tea_cls_scores:", tea_cls_scores[i].shape)#[1,10000,256]
        #     print("tea_bbox_preds:", tea_bbox_preds[i].shape)#[1,4,100/50/25/13/7,100/50/25/13/7]
        #     print("tea_centernesses:", tea_centernesses[i].shape)#[1,1,100/50/25/13/7,100/50/25/13/7]


        #-------------------------------------------------------
        enhanced_text = []  # 保存由视觉上下文反向增强后的文本特征。
        for i, feat in enumerate(stu_x):  # 循环处理 batch、FPN 层或类别维度，保持多尺度逻辑一致。
            enhanced = self.text_enhancer(text_embeds,feat )
            enhanced_text.append(enhanced)

        reused_cls_feat=enhanced_text  # 分类分支复用视觉增强文本，模拟教师视觉语言分类头输入。
        reused_reg_feat=fused_feats  # 回归分支复用跨模态融合视觉特征，模拟教师回归头输入。
        reused_cls_scores, reused_bbox_preds, reused_centernesses = \
            self.reuse_teacher_head1(reused_reg_feat,
                                     reused_cls_feat
                                     )
        # for i in range(len(reused_cls_scores)):
        #     print("reused_cls_scores:", reused_cls_scores[i].shape)  # [1,10000,256]
        #     print("reused_bbox_preds:", reused_bbox_preds[i].shape)  # [1,4,100/50/25/13/7,100/50/25/13/7]
        #     print("reused_centernesses:", reused_centernesses[i].shape)  # [1,1,100/50/25/13/7,100/50/25/13/7]
        #     # 执行可视化
        # visualize_features(
        #     reused_cls_scores,
        #     reused_bbox_preds,
        #     reused_centernesses,
        #     mode='max',  # 可选 'pca' 或 'max'
        #     target_size=300,  # 统一尺寸
        #     save_path='./feature_visualization_reused'  # 保存路径
        # )
        # #----------------------
        # analyze_feature_spaces_pca_all_levels(stu_x, tea_x, fused_feats, levels=None)

        outputs = unpack_gt_instances(batch_data_samples)  # 拆分 batch 标注，准备 ATSS 目标分配和损失计算。
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs
        # ====== add by szz 在 loss_by_feat 调用之前加 ======
        # 把 class_pos_map 和 stu_cls_flat 缓存为实例属性，避免修改 loss_by_feat 签名
        self._kd_class_pos_map = self._build_class_pos_map(  # 缓存类别到文本 token 的正样本映射，用于把 token-space 分类 KD 转到 class-space。
            batch_data_samples,
            num_classes=self.bbox_head.num_classes,
            device=batch_inputs.device)
        # stu_cls_scores 在 loss() 里已经有了
                # for i in range(len(stu_cls_scores)):
        #     print("fused_feats",stu_cls_scores[i].shape) # torch.Size([1, 80, 100, 100])
        #     print("fused_feats",  stu_bbox_preds[i].shape)  # torch.Size([1, 4, 100, 100])
        #     print("stu_centernesses:",stu_centernesses[i].shape) #stu_centernesses: torch.Size([1, 1, 100, 100])
        stu_cls_flat_list = []
        for lvl in stu_cls_scores:  # 循环处理 batch、FPN 层或类别维度，保持多尺度逻辑一致。
            B_s, C_s, H_s, W_s = lvl.shape
            stu_cls_flat_list.append(lvl.permute(0, 2, 3, 1).reshape(B_s, -1, C_s))
        self._kd_stu_cls_flat = torch.cat(stu_cls_flat_list, dim=1)  # [B, 13343, 6]
        # ======================================
        losses = self.loss_by_feat(  # 进入核心损失函数，同时计算检测损失和增强蒸馏损失。
                                   tea_cls_scores,
                                   tea_bbox_preds,
                                   tea_centernesses,
                                   tea_x,
                                   tea_reg_hold, #add
                                   stu_cls_scores,
                                   stu_bbox_preds,
                                   stu_centernesses,
                                   stu_x,
                                   fused_feats,
                                   text_embeds, #add
                                   proj_text,
                                   reused_cls_scores,
                                   reused_bbox_preds,
                                   reused_centernesses,
                                   batch_gt_instances,
                                   batch_img_metas,
                                   batch_gt_instances_ignore)
        # print("losses:",losses)
        return losses  # 返回训练框架需要反向传播的所有检测损失与 KD 损失。
    #'''对某一层 FPN 特征 x，分别经过 ATSS 检测头的分类分支和回归分支，得到分类预测、边界框预测、中心度预测，同时在中间层截取一份特征用于蒸馏。'''
    def forward_hkd_single(self, x, scale, module):  # 定义 forward_hkd_single 函数/方法，承担当前蒸馏流程中的一个子步骤。
        cls_feat, reg_feat = x, x  # 分类和回归分支从同一层 FPN 特征开始分流。
        cls_feat_hold, reg_feat_hold = x, x
        for i, cls_conv in enumerate(module.bbox_head.cls_convs):  # 循环处理 batch、FPN 层或类别维度，保持多尺度逻辑一致。
            cls_feat = cls_conv(cls_feat, activate=False)
            if i + 1 == self.reused_teacher_head_idx:  # 条件分支，根据当前状态选择不同处理路径。
                #cls_feat: [B, 256, H, W]  从 FPN 特征中提取更适合做分类的高级语义特征。 空间尺寸和通道数通常保持不变，但语义被分类任务重塑了。
                #在分类/回归卷积塔的某一层中间位置截取特征，保存下来用于蒸馏或复用教师头。
                cls_feat_hold = cls_feat

            cls_feat = cls_conv.activate(cls_feat)

        for i, reg_conv in enumerate(module.bbox_head.reg_convs):  # 循环处理 batch、FPN 层或类别维度，保持多尺度逻辑一致。
            reg_feat = reg_conv(reg_feat, activate=False)
            if i + 1 == self.reused_teacher_head_idx:  # 条件分支，根据当前状态选择不同处理路径。
                reg_feat_hold = reg_feat
            reg_feat = reg_conv.activate(reg_feat)
        #是 最终分类预测层。 它通常是一个简单的卷积层：  cls_score: [B, 6, H, W]
        cls_score = module.bbox_head.atss_cls(cls_feat)  # 分类头输出学生/模块的类别 logits。
        #是 最终边界框回归预测层。 bbox_pred: [B, num_base_priors * 4, H, W]  4表示四个点坐标
        #> scale为不同 FPN 层的预测框尺度差异很大，所以为每一层学习一个独立缩放因子，使 bbox 回归数值更稳定。

        bbox_pred = scale(module.bbox_head.atss_reg(reg_feat)).float()#atss head
        centerness = module.bbox_head.atss_centerness(reg_feat)  # 中心度头预测 anchor/位置位于目标中心附近的质量分数。
      #
        for i in range(len(cls_score)):  # 循环处理 batch、FPN 层或类别维度，保持多尺度逻辑一致。
            assert torch.isfinite(cls_score[i]).all(), "s-logits-"  # 数值/结构检查，尽早发现 NaN、Inf 或维度不匹配问题。
            assert torch.isfinite(bbox_pred[i]).all(), "s-reg-"  # 数值/结构检查，尽早发现 NaN、Inf 或维度不匹配问题。
            assert torch.isfinite(centerness[i]).all(), "s-center-"  # 数值/结构检查，尽早发现 NaN、Inf 或维度不匹配问题。
        return cls_score, bbox_pred, centerness, cls_feat_hold, reg_feat_hold  # 返回当前步骤的结果，供上层训练流程继续使用。


    def _apply_dapc_to_caption_tokens(
        self,
        embedding: Tensor,
        class_positive_map: Optional[Tensor],
    ) -> Tensor:
        """Inject DAPC prototypes into the real caption token slots.

        ``class_positive_map[k, j] > 0`` means caption token slot ``j``
        belongs to class ``k``. This avoids the earlier 0..K slot assumption
        and keeps GLIP's token-level alignment with ``positive_maps``.
        """
        if (not getattr(self, 'enable_dapc', False)
                or class_positive_map is None
                or not hasattr(self, 'dapc_prototypes')):
            return embedding

        alpha = float(getattr(self, 'dapc_alpha', 0.35))
        if alpha <= 0:
            return embedding

        dapc_embedding = embedding.clone()
        num_classes = min(class_positive_map.shape[0], self.dapc_prototypes.shape[0])
        for k in range(num_classes):
            token_inds = torch.nonzero(
                class_positive_map[k].to(embedding.device) > 0,
                as_tuple=True)[0]
            if token_inds.numel() == 0:
                continue

            proto_len = int(self.dapc_token_lengths[k].item())
            proto = self.dapc_prototypes[k, :proto_len].to(embedding.device)
            if proto.shape[0] == token_inds.numel():
                token_proto = proto
            else:
                token_proto = proto.mean(0, keepdim=True).expand(token_inds.numel(), -1)

            original = embedding[:, token_inds, :]
            injected = (1.0 - alpha) * original + alpha * token_proto.unsqueeze(0)
            dapc_embedding[:, token_inds, :] = F.normalize(injected, p=2, dim=-1)

        return dapc_embedding

    def forward_crosskd_single_tea(self, visual_feats, batch_inputs,batch_data_samples,module):  # 定义 forward_crosskd_single_tea 函数/方法，承担当前蒸馏流程中的一个子步骤。
        # add by jojo
        text_prompts = ['crazing. inclusion. patches. pitted surface. rolled-in scale. scratches']  # 构造教师语言模型的类别文本提示，决定文本 token 与类别语义。
        # text_prompts = ['huahen. liebian. huanzhangshamu. wuzhuangshamu. xiaoquanliebian. xiaomianbuliang']
        # text_prompts = ['crazing. inclusion. patches. pitted surface. rolled-in scale. scratches']
        # text_prompts = [
        #     data_samples.text for data_samples in batch_data_samples
        # ]            #这里从 batch 里的每张图取出 GT 类别。后面要构建 positive_map。
        gt_labels = [  # 读取每张图的 GT 类别，用于从文本 token 中抽取正样本片段。
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples  # 循环处理 batch、FPN 层或类别维度，保持多尺度逻辑一致。
        ]
        # print(" gt_labels:", gt_labels)

        new_text_prompts = []
        positive_maps = []  # 保存每张图 GT 类别对应的文本 token 正区域映射。
        dapc_class_positive_map = None
        if len(set(text_prompts)) == 1:  # 条件分支，根据当前状态选择不同处理路径。  如果 batch 中所有图片使用同一个 prompt，就没必要重复 tokenize。
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            #这一步把文本 prompt 变成 tokenizer 能处理的形式。
            #tokens_positive 的作用就是记录每个类别对应哪些 token 位置。
            tokenized, caption_string, tokens_positive, _ = \
                module.get_tokens_and_prompts(
                    text_prompts[0], True)
            new_text_prompts = [caption_string] * len(batch_inputs)
            # Full class-to-token map for DAPC injection. Unlike per-image
            # positive_maps below, this keeps all K class token positions in
            # the original caption and is aligned with GLIP token slots.
            _, dapc_class_positive_map = module.get_positive_map(
                tokenized, tokens_positive)
            # print("new_text_prompts", new_text_prompts)

            for gt_label in gt_labels:  # 循环处理 batch、FPN 层或类别维度，保持多尺度逻辑一致。
                new_tokens_positive = [
                    tokens_positive[label] for label in gt_label
                ]
                _, positive_map = module.get_positive_map(
                    tokenized, new_tokens_positive)
#假设某张图有两个 GT 类别：gt_label = tensor([0, 5])也就是：crazing、scratches 然后 get_positive_map() 会生成一个矩阵，表示：new_tokens_positive = [
 # tokens_positive[0],
 # tokens_positive[5]
#]
#> 当前图像中每个 GT box 对应文本 prompt 中哪些 token 是正样本。
#通常形状类似：positive_map: [num_gt, max_text_len]
                positive_maps.append(positive_map)
        else:  # 兜底分支，处理上述条件都不满足的情况。
            for text_prompt, gt_label in zip(text_prompts, gt_labels):  # 循环处理 batch、FPN 层或类别维度，保持多尺度逻辑一致。
                tokenized, caption_string, tokens_positive, _ = \
                    module.get_tokens_and_prompts(
                        text_prompt, True)
                new_tokens_positive = [
                    tokens_positive[label] for label in gt_label
                ]
                _, positive_map =module.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
                new_text_prompts.append(caption_string)
#它把每张图对应的 positive_map 存到：data_samples.gt_instances.positive_maps  后面你的代码里会用到它：self._kd_class_pos_map = self._build_class_pos_map(...)
#会从 batch_data_samples 里面读取：然后聚合成：class_pos_map: [num_classes, 256]  做分类KD

        for i, data_samples in enumerate(batch_data_samples):  # 循环处理 batch、FPN 层或类别维度，保持多尺度逻辑一致。
            # .bool().float() is very important
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            data_samples.gt_instances.positive_maps = positive_map

#embedded 是文本 token 的嵌入；
#masks 表示哪些 token 是有效的，哪些是 padding；
#256 是最大 token 长度；
#768 是 BERT/语言模型隐藏维度。
        language_feats  = module.language_model(new_text_prompts)  
        # 教师语言模型编码文本提示，得到 embedded/masks 等语言特征。
#这里把视觉特征和语言特征组装成一个字典：这个 feat_inputs 后面会送到：module.bbox_head.head.dyhead_tower(feat_inputs)也就是 DyHead 或视觉语言融合塔。
        feat_inputs = {"visual":  visual_feats ,
                       "lang": language_feats}
                       #也就是说，分类 KD 时只比较有效 token，不比较 padding token。
        self.text_masks = language_feats['masks']  # 缓存有效文本 token mask，后续分类蒸馏时屏蔽 padding token。
        #--------此处无问题
        #这里不是普通 ATSS 的：说明教师模型的 head 是一个 视觉-语言检测头。教师检测头前向
        bbox_pred, centerness, cls_score = module.bbox_head.head(visual_feats,  # 教师检测头融合视觉与语言特征，输出蒸馏目标预测。
                                                       language_feats)

        for i in range(len(cls_score)):  # 循环处理 batch、FPN 层或类别维度，保持多尺度逻辑一致。
            #教师的分类预测。 但这里的分类预测不是传统类别数，而是文本 token 维度。  代表：> 每个视觉位置/anchor 与每个文本 token 的匹配分数。
            assert torch.isfinite(cls_score[i]).all(), "t-cls-logits"  # 数值/结构检查，尽早发现 NaN、Inf 或维度不匹配问题。
            assert torch.isfinite(bbox_pred[i]).all(), "t-reg_"  # 数值/结构检查，尽早发现 NaN、Inf 或维度不匹配问题。
            assert torch.isfinite(centerness[i]).all(), "t-center-"  # 数值/结构检查，尽早发现 NaN、Inf 或维度不匹配问题。


        #-------done!
        #获取教师文本嵌入 embedding  这里根据 early_fuse 判断使用哪种文本特征。
        if module.bbox_head.head.early_fuse:  # 条件分支，根据当前状态选择不同处理路径。  此时得到的 embedding 不是原始文本特征，而是：经过视觉-语言交互后的文本特征
            #这对蒸馏很重要，因为它比纯文本 embedding 更贴近当前图像。
            embedding = module.bbox_head.head.dyhead_tower(feat_inputs)['lang']['hidden']  # text-visual pairs
            # print("early fuse!") #enter this!
        else:  # 兜底分支，处理上述条件都不满足的情况。
            embedding = language_feats['embedded']  # text-visual pairs
#也就是每个 token 的 768 维向量被归一化为单位长度。
#为什么要归一化？因为后续文本向量通常会参与点积分类：visual_feature · text_embedding 如果不归一化，向量模长会影响 logits 大小，不稳定。
        embedding = F.normalize(embedding, p=2, dim=-1)  # [B, 256, 768]
        embedding = self._apply_dapc_to_caption_tokens(
            embedding, dapc_class_positive_map)

        #提取教师 DyHead 视觉特征
        reg_feat_hold = []  # 保存教师 DyHead 视觉特征，作为回归蒸馏中间知识。
        for i, feature in enumerate(visual_feats):
            reg_feat0 = module.bbox_head.head.dyhead_tower(feat_inputs)['visual'][i]
            reg_feat_hold.append(reg_feat0)
        cls_feat_hold = embedding
                # print("cla_feat_hold:", cls_feat_hold.shape)  # [2,256,768]
            # reg_feat: torch.Size([2, 256, 100, 100])
            # reg_feat: torch.Size([2, 256, 50, 50])
            # torch.Size([2, 256, 25, 25])
            # torch.Size([2, 256, 13, 13])
            # torch.Size([2, 256, 7, 7])
#cls_feat_hold=embedding
#这里把教师文本 embedding 作为 cls_feat_hold 返回。
#在 loss() 里：
#text_embeds = tea_cls_hold
#所以：
#tea_cls_hold == cls_feat_hold == embedding
#也就是说，后续学生分支中的：
#text_embeds = tea_cls_hold
#proj_text = self.text_proj(text_embeds)
#用的就是教师文本特征。
#提供教师预测结果，作为 KD target  后面在 loss_by_feat() 中会和学生复用教师头的输出比较：
        return cls_score, bbox_pred, centerness,cls_feat_hold, reg_feat_hold  # 返回当前步骤的结果，供上层训练流程继续使用。

    def reuse_teacher_head1(self, reused_reg_feat,reused_cls_feat  # 定义 reuse_teacher_head1 函数/方法，承担当前蒸馏流程中的一个子步骤。
                            ):

        bbox_preds = []
        centerness = []  # 中心度头预测 anchor/位置位于目标中心附近的质量分数。
        cls_logits = []
        MAX_CLAMP_VALUE = 50000
        for i in range(len(reused_reg_feat)):  # 循环处理 batch、FPN 层或类别维度，保持多尺度逻辑一致。
            visual = reused_reg_feat[i]
            embedding = reused_cls_feat[i]

            dot_product_proj_tokens = self.teacher.bbox_head.head.dot_product_projection_text(embedding /  # 将增强文本映射到教师分类头的 token 投影空间。
                                                                                              2.0)
            dot_product_proj_tokens_bias = torch.matmul(
                embedding, self.teacher.bbox_head.head.bias_lang) + self.teacher.bbox_head.head.bias0

            B, C, H, W = visual.shape

            bbox_pred = self.teacher.bbox_head.head.scales[i](self.teacher.bbox_head.head.bbox_pred(visual))
            bbox_preds.append(bbox_pred)
            centerness.append(self.teacher.bbox_head.head.centerness(visual))

            dot_product_proj_queries = permute_and_flatten(  # 展平视觉特征为每个 anchor/位置的查询向量。
                visual, B, self.teacher.bbox_head.head.num_base_priors, C, H, W)

            bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(
                1, self.teacher.bbox_head.head.num_base_priors, 1)
            dot_product_logit = (  # 视觉查询与文本 token 做点积，形成 open-vocabulary 风格分类 logits。
                                        torch.matmul(dot_product_proj_queries,
                                                     dot_product_proj_tokens.transpose(-1, -2)) /
                                        self.teacher.bbox_head.head.log_scale.exp()) + bias
            dot_product_logit = torch.clamp(  # 视觉查询与文本 token 做点积，形成 open-vocabulary 风格分类 logits。
                dot_product_logit, max=MAX_CLAMP_VALUE)
            dot_product_logit = torch.clamp(  # 视觉查询与文本 token 做点积，形成 open-vocabulary 风格分类 logits。
                dot_product_logit, min=-MAX_CLAMP_VALUE)
            cls_logits.append(dot_product_logit)

        reused_cls_score = cls_logits
        reused_bbox_pred = bbox_preds
        reused_centerness = centerness
        return reused_cls_score, reused_bbox_pred, reused_centerness  # 返回当前步骤的结果，供上层训练流程继续使用。

    #-----add by jojo 20250606--------
    def dist2(self,tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None):  # 定义 dist2 函数/方法，承担当前蒸馏流程中的一个子步骤。
        diff = (tensor_a - tensor_b) ** 2
        #   print(diff.size())      batchsize x 1 x W x H,
        #   print(attention_mask.size()) batchsize x 1 x W x H
        diff = diff * attention_mask
        diff = diff * channel_attention_mask
        diff = torch.sum(diff) ** 0.5
        return diff  # 返回当前步骤的结果，供上层训练流程继续使用。
    #-------add end-------------------

    def loss_by_feat(  # 定义 loss_by_feat 函数/方法，承担当前蒸馏流程中的一个子步骤。
            self,
            tea_cls_scores: List[Tensor],
            tea_bbox_preds: List[Tensor],
            tea_centernesses: List[Tensor],
            tea_feats: List[Tensor],
            tea_reg_hold,  # add
            cls_scores: List[Tensor],#stu
            bbox_preds: List[Tensor],
            centernesses: List[Tensor],
            feats: List[Tensor],
            fused_feats,
            t_text, #add by jojo
            s_text,#add by jojo
            reused_cls_scores,
            reused_bbox_preds,
            reused_centernesses,
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # 获取当前KD权重
        kd_weights = self.kd_weight_scheduler  # 读取当前动态蒸馏权重，控制分类/回归/中心度 KD 对总损失的贡献。
       # # # ---------this is for atss head---------
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]  # 统计学生各 FPN 层尺寸，用于生成 anchors 和 ATSS 目标。
        assert len(featmap_sizes) == self.bbox_head.prior_generator.num_levels  # 数值/结构检查，尽早发现 NaN、Inf 或维度不匹配问题。

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.bbox_head.get_anchors(  # 根据特征图尺寸生成每张图每层 anchor 及有效标记。
            featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = self.bbox_head.get_targets(  # ATSS 根据 GT 自适应分配正负样本并生成分类/回归目标。
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        # self.bbox_head.get_targets主要用于 ​​动态分配正负样本​​，
        #解决目标检测中因锚点（Anchor）或锚点自由（Anchor - Free）方法导致的样本不均衡问题。在
        # ATSSHead中，该方法会根据自适应策略筛选高质量的正样本，而非依赖固定 IoU阈值。

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor) = cls_reg_targets

        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=device)).item()

        #srudent branch
        losses_cls, losses_bbox, loss_centerness, \
            bbox_avg_factor = multi_apply(
                self.bbox_head.loss_by_feat_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                centernesses,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                avg_factor=avg_factor)

        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        losses = dict(loss_cls=losses_cls, loss_bbox=losses_bbox,loss_centerness=loss_centerness)  # 先组织常规检测损失，后续再追加 KD 损失。
       #  # --------------end
#++++++++++++
#第一，针对 reused_bbox_preds 的尺寸生成 teacher anchors/targets  不过注意，你当前后面并没有明显使用：
#也就是说，这部分可能是旧版 KD 逻辑遗留，或者给 pred_imitation_loss_single() 用的，但当前主要 KD 是：calculate_enhanced_kd_loss()
        featmap_sizes1 = [featmap.size()[-2:] for featmap in reused_bbox_preds]
        assert len(featmap_sizes1) == self.teacher.bbox_head.prior_generator.num_levels  # 数值/结构检查，尽早发现 NaN、Inf 或维度不匹配问题。

        device1 = reused_cls_scores[0].device
        anchor_list1, valid_flag_list1 = self.teacher.bbox_head.get_anchors(
            featmap_sizes1, batch_img_metas, device=device1)

        cls_reg_targets1 = self.teacher.bbox_head.get_targets(
            anchor_list1,
            valid_flag_list1,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        (anchor_list1, labels_list1, label_weights_list1, bbox_targets_list1,
         bbox_weights_list1, avg_factor1) = cls_reg_targets1
        avg_factor1 = reduce_mean(
            torch.tensor(avg_factor1, dtype=torch.float, device=device1)).item()

        anchors1 = torch.cat(anchor_list1, dim=1)
        labels1 = torch.cat(labels_list1, dim=1)
        label_weights1 = torch.cat(label_weights_list1, dim=1)
        bbox_targets1 = torch.cat(bbox_targets_list1, dim=1)
#++++++++++++6.2 第二部分：准备 KD 的 teacher/reused 输出
#第二，把多层 FPN 分类输出拼接起来
        tea_cls_scores = torch.cat(tea_cls_scores, dim=1)  ##[2,13343,256]
        # for i in range(len(reused_cls_scores)):
        #这里 13343 是所有 FPN 层空间位置数加起来，例如：10000 + 2500 + 625 + 169 + 49 = 13343
        #     print("resued_cls_scores:", reused_cls_scores[i].shape) [2,10000/2500/625/169,256]
        reused_cls_scores = torch.cat(reused_cls_scores, dim=1)  # [2,13343,256]


        # for i in range(len(anchors1)):
        #     print("anchors1:", anchors1[i].shape)#[13343,4]
        #     print("labels1:", labels1[i].shape)#[13343,256]
        #     print("label_weights1", label_weights1[i].shape)#[13343]
        #     print("bbox_targets1", bbox_targets1[i].shape)#[13343,4]

        tea_centernesses_ = []
        tea_bbox_preds_ = []
        reused_centernesses_ = []
        reused_bbox_preds_ = []
        #6.3 第三部分：把 bbox/centerness 也展平拼接
        #每一层原来是 bbox_pred: [B, 4, H, W]   center:    [B, 1, H, W]
        #转换成： bbox_pred: [B, H*W, 4]  center:    [B, H*W, 1]
        #最后拼接起来 tea_bbox_preds:        [B, 13343, 4] tea_centernesses:      [B, 13343, 1]
        #reused_bbox_preds:     [B, 13343, 4]   reused_centernesses:   [B, 13343, 1]
        for tea_bbox_pred, tea_centerness, reused_bbox_pred, reused_centerness in (  # 循环处理 batch、FPN 层或类别维度，保持多尺度逻辑一致。
                zip(tea_bbox_preds, tea_centernesses, reused_bbox_preds, reused_centernesses)):
            tea_centernesses_.append(
                tea_centerness.permute(0, 2, 3,
                                       1).reshape(tea_cls_scores.size(0), -1, 1))
            tea_bbox_preds_.append(
                tea_bbox_pred.permute(0, 2, 3,
                                      1).reshape(tea_cls_scores.size(0), -1, 4))
            reused_centernesses_.append(
                reused_centerness.permute(0, 2, 3,
                                          1).reshape(reused_cls_scores.size(0), -1, 1))
            reused_bbox_preds_.append(
                reused_bbox_pred.permute(0, 2, 3,
                                         1).reshape(reused_cls_scores.size(0), -1, 4))
        tea_bbox_preds = torch.cat(tea_bbox_preds_, dim=1)
        tea_centernesses = torch.cat(tea_centernesses_, dim=1)
        reused_bbox_preds = torch.cat(reused_bbox_preds_, dim=1)  # torch.Size([2, 13343, 4])
        reused_centernesses = torch.cat(reused_centernesses_, dim=1)  # torch.Size([2, 13343, 1])

        # 2. 计算改进的KD损失（原来的调用，加两个新参数）
        kd_losses = self.calculate_enhanced_kd_loss(  # 计算增强 KD：分类 token-to-class、回归框、中心度三类蒸馏。
            tea_cls_scores, tea_bbox_preds, tea_centernesses,  # 教师前向得到分类 token logits、框回归和中心度，同时保留文本/回归中间特征。
            reused_cls_scores, reused_bbox_preds, reused_centernesses,  # 把学生增强特征送入教师头结构，生成用于和教师输出做 KD 的复用预测。
            kd_weights, device
        )

        # # 2. 计算改进的KD损失
        # kd_losses = self.calculate_enhanced_kd_loss(
        #     tea_cls_scores, tea_bbox_preds, tea_centernesses,
        #     reused_cls_scores, reused_bbox_preds, reused_centernesses,
        #     kd_weights, device
        # )
        losses.update(kd_losses)  # 把 KD 损失并入总损失字典参与反向传播。

        # 3. 修复梯度统计计算
        try:  # 保护性执行，避免统计或蒸馏异常中断训练。
            # 正确提取损失值
            det_loss_value = 0.0
            for k in ['loss_cls', 'loss_bbox', 'loss_centerness']:  # 循环处理 batch、FPN 层或类别维度，保持多尺度逻辑一致。
                if k in losses:  # 条件分支，根据当前状态选择不同处理路径。
                    # 确保是张量然后获取数值
                    if isinstance(losses[k], torch.Tensor):  # 条件分支，根据当前状态选择不同处理路径。
                        det_loss_value += losses[k].item()
                    elif isinstance(losses[k], (int, float)):  # 追加条件分支，处理另一种训练或数据状态。
                        det_loss_value += losses[k]

            kd_loss_value = 0.0
            for k in losses:  # 循环处理 batch、FPN 层或类别维度，保持多尺度逻辑一致。
                if 'kd' in k:  # 条件分支，根据当前状态选择不同处理路径。
                    if isinstance(losses[k], torch.Tensor):  # 条件分支，根据当前状态选择不同处理路径。
                        kd_loss_value += losses[k].item()
                    elif isinstance(losses[k], (int, float)):  # 追加条件分支，处理另一种训练或数据状态。
                        kd_loss_value += losses[k]

            # 更新梯度统计（使用指数移动平均）
            self.gradient_stats['det_loss'] = (  # 初始化/更新 self.gradient_stats['det_loss']，作为模型状态或蒸馏模块配置。
                    0.9 * self.gradient_stats['det_loss'] + 0.1 * det_loss_value
            )
            self.gradient_stats['kd_loss'] = (  # 初始化/更新 self.gradient_stats['kd_loss']，作为模型状态或蒸馏模块配置。
                    0.9 * self.gradient_stats['kd_loss'] + 0.1 * kd_loss_value
            )

        except Exception as e:  # 捕获异常并给出安全回退，保证训练流程继续。
            print(f"Gradient statistics error: {e}")
            # 设置默认值保证训练继续
            self.gradient_stats['det_loss'] = 1.0  # 初始化/更新 self.gradient_stats['det_loss']，作为模型状态或蒸馏模块配置。
            self.gradient_stats['kd_loss'] = 0.5  # 初始化/更新 self.gradient_stats['kd_loss']，作为模型状态或蒸馏模块配置。

        return losses  # 返回训练框架需要反向传播的所有检测损失与 KD 损失。

# 原始
    # def calculate_enhanced_kd_loss(self, tea_cls, tea_bbox, tea_center,
    #                                reused_cls, reused_bbox, reused_center,
    #                                kd_weights, device):
    #     """改进的KD损失计算"""
    #     losses = {}

    #     try:
    #         # 1. 自适应温度缩放的分类KD损失
    #         temperature = max(0.5, 4.0 * (1 - self.current_epoch / self.total_epochs))

    #         cls_loss = 0
    #         """遍历的是 batch 维，所以每次 t, s 的 shape 是 [13343, 256]，在256个token上做 softmax + KL，大量无关 token 稀释，结果趋近于 0。"""
    #         for t, s in zip(tea_cls, reused_cls):
    #             # 确保维度匹配
    #             if t.shape == s.shape:
    #                 # 使用温度缩放的KL散度
    #                 t_soft = F.softmax(t / temperature, dim=-1)
    #                 s_soft = F.log_softmax(s / temperature, dim=-1)
    #                 cls_loss += F.kl_div(s_soft, t_soft, reduction='batchmean') * (temperature ** 2)

    #         if cls_loss > 0:
    #             losses['loss_cls_kd'] = cls_loss / len(tea_cls) * kd_weights['cls']

    #         # 2. 焦点回归KD损失
    #         reg_loss = 0
    #         for t, s in zip(tea_bbox, reused_bbox):
    #             if t.shape == s.shape:
    #                 # 使用smooth L1损失，对困难样本加权
    #                 diff = torch.abs(t - s)
    #                 weight = torch.exp(diff)  # 困难样本权重更高
    #                 reg_loss += (F.smooth_l1_loss(s, t, reduction='none') * weight).mean()

    #         if reg_loss > 0:
    #             losses['loss_reg_kd'] = reg_loss / len(tea_bbox) * kd_weights['reg']

    #         # 3. 中心度KD损失
    #         center_loss = 0
    #         for t, s in zip(tea_center, reused_center):
    #             if t.shape == s.shape:
    #                 center_loss += F.binary_cross_entropy_with_logits(
    #                     s, t.sigmoid(), reduction='mean')

    #         if center_loss > 0:
    #             losses['loss_center_kd'] = center_loss / len(tea_center) * kd_weights['center']

    #     except Exception as e:
    #         print(f"Enhanced KD loss error: {e}")
    #         # 保证返回有效的损失张量
    #         for key in ['loss_cls_kd', 'loss_reg_kd', 'loss_center_kd']:
    #             losses[key] = torch.tensor(0.0, device=device)

    #     return losses
    # ============================================================
    #     # add by szz
    # def _build_class_pos_map(self, batch_data_samples, num_classes, device):
    #     """从 batch gt 里聚合 class-level positive_map [num_classes, 256]"""
    #     class_pos_map = torch.zeros(num_classes, 256, device=device)
    #     for data_sample in batch_data_samples:
    #         gt = data_sample.gt_instances
    #         if (hasattr(gt, 'positive_maps')
    #                 and hasattr(gt, 'labels')
    #                 and len(gt.labels) > 0):
    #             pm = gt.positive_maps.to(device).float()   # [num_gt, 256]
    #             lbls = gt.labels.to(device)                # [num_gt]
    #             for c in range(num_classes):
    #                 mask = (lbls == c)
    #                 if mask.any():
    #                     class_pos_map[c] = (
    #                         class_pos_map[c] + pm[mask].sum(0)).clamp(0, 1)
    #     return class_pos_map  # [num_classes, 256]
    # # add by szz
    # def calculate_enhanced_kd_loss(self, tea_cls, tea_bbox, tea_center,
    #                                 reused_cls, reused_bbox, reused_center,
    #                                 kd_weights, device):   # ← 签名不变！
    #     losses = {}
    #     try:
    #         temperature = max(1.0, 4.0 * (1 - self.current_epoch / self.total_epochs))

    #         # ============================================================
    #         # 1. 分类 KD：class-space BCE
    #         # ============================================================
    #         class_pos_map = getattr(self, '_kd_class_pos_map', None)
    #         student_cls_flat = getattr(self, '_kd_stu_cls_flat', None)

    #         if (class_pos_map is not None
    #                 and student_cls_flat is not None
    #                 and class_pos_map.sum() > 0):

    #             B, N, T = tea_cls.shape
    #             num_classes = class_pos_map.shape[0]

    #             # token logits → 6类 logits（max pooling over relevant tokens）
    #             tea_cls_6 = torch.zeros(B, N, num_classes,
    #                                     device=tea_cls.device,
    #                                     dtype=tea_cls.dtype)
    #             for c in range(num_classes):
    #                 token_mask = class_pos_map[c] > 0.5
    #                 if token_mask.sum() > 0:
    #                     tea_cls_6[:, :, c] = tea_cls[:, :, token_mask].max(dim=-1).values

    #             tea_soft = torch.sigmoid(tea_cls_6 / temperature).detach()  # [B, N, 6]

    #             # 只对 teacher 有响应的前景 anchor 计算
    #             fg_mask = tea_soft.max(dim=-1).values > 0.10  # [B, N]
    #             if fg_mask.sum() > 0:
    #                 stu_sel = student_cls_flat[fg_mask]   # [num_fg, 6]
    #                 tea_sel = tea_soft[fg_mask]           # [num_fg, 6]
    #                 losses['loss_cls_kd'] = F.binary_cross_entropy_with_logits(
    #                     stu_sel / temperature,
    #                     tea_sel,
    #                     reduction='mean'
    #                 ) * (temperature ** 2) * kd_weights.get('cls', 1.0)
    #             else:
    #                 losses['loss_cls_kd'] = student_cls_flat.sum() * 0.0
    #         else:
    #             # positive_map 未就绪时的 fallback
    #             losses['loss_cls_kd'] = tea_cls.sum() * 0.0

    #         # ============================================================
    #         # 2. 回归 KD（原逻辑保持不变）
    #         # ============================================================
    #         reg_loss = 0
    #         for t, s in zip(tea_bbox, reused_bbox):
    #             if t.shape == s.shape:
    #                 diff = torch.abs(t - s)
    #                 weight = torch.exp(diff)
    #                 reg_loss += (F.smooth_l1_loss(s, t, reduction='none') * weight).mean()
    #         if reg_loss > 0:
    #             losses['loss_reg_kd'] = reg_loss / len(tea_bbox) * kd_weights.get('reg', 1.0)

    #         # ============================================================
    #         # 3. 中心度 KD（原逻辑保持不变）
    #         # ============================================================
    #         center_loss = 0
    #         for t, s in zip(tea_center, reused_center):
    #             if t.shape == s.shape:
    #                 center_loss += F.binary_cross_entropy_with_logits(
    #                     s, t.sigmoid(), reduction='mean')
    #         if center_loss > 0:
    #             losses['loss_center_kd'] = center_loss / len(tea_center) * kd_weights.get('center', 1.0)

    #     except Exception as e:
    #         print(f"Enhanced KD loss error: {e}")
    #         for key in ['loss_cls_kd', 'loss_reg_kd', 'loss_center_kd']:
    #             if key not in losses:
    #                 losses[key] = torch.tensor(0.0, device=device, requires_grad=False)

    #     return losses
    # ============================================================
        # add by szz
    def _build_class_pos_map(self, batch_data_samples, num_classes, device):  # 定义 _build_class_pos_map 函数/方法，承担当前蒸馏流程中的一个子步骤。
        """从 batch gt 里聚合 class-level positive_map [num_classes, 256]"""
        class_pos_map = torch.zeros(num_classes, 256, device=device)  # 初始化类别到文本 token 的聚合正样本矩阵。
        for data_sample in batch_data_samples:  # 循环处理 batch、FPN 层或类别维度，保持多尺度逻辑一致。
            gt = data_sample.gt_instances
            if (hasattr(gt, 'positive_maps')  # 条件分支，根据当前状态选择不同处理路径。
                    and hasattr(gt, 'labels')
                    and len(gt.labels) > 0):
                pm = gt.positive_maps.to(device).float()   # [num_gt, 256]
                lbls = gt.labels.to(device)                # [num_gt]
                for c in range(num_classes):  # 循环处理 batch、FPN 层或类别维度，保持多尺度逻辑一致。
                    mask = (lbls == c)
                    if mask.any():  # 条件分支，根据当前状态选择不同处理路径。
                        class_pos_map[c] = (
                            class_pos_map[c] + pm[mask].sum(0)).clamp(0, 1)
        return class_pos_map  # [num_classes, 256]
    # add by szz
    def calculate_enhanced_kd_loss(self, tea_cls, tea_bbox, tea_center,  # 定义 calculate_enhanced_kd_loss 函数/方法，承担当前蒸馏流程中的一个子步骤。
                                    reused_cls, reused_bbox, reused_center,
                                    kd_weights, device):   # ← 签名不变！
        losses = {}
        try:  # 保护性执行，避免统计或蒸馏异常中断训练。
            temperature = max(1.0, 4.0 * (1 - self.current_epoch / self.total_epochs))  # 温度系数控制教师软标签平滑程度，训练越后期通常越接近 1。

            # ============================================================
            # 1. 分类 KD：class-space BCE
            # ============================================================
            class_pos_map = getattr(self, '_kd_class_pos_map', None)

            if (class_pos_map is not None  # 条件分支，根据当前状态选择不同处理路径。
                    and class_pos_map.sum() > 0):

                B, N, T = tea_cls.shape
                num_classes = class_pos_map.shape[0]

                # token logits → 6类 logits（max pooling over relevant tokens）
                tea_cls_6 = torch.zeros(B, N, num_classes,  # 把教师 token logits 聚合到数据集类别空间。
                                        device=tea_cls.device,
                                        dtype=tea_cls.dtype)
                reused_cls_6 = torch.zeros(B, N, num_classes,  # 把复用头 token logits 同样聚合到类别空间以便和教师对齐。
                                           device=reused_cls.device,
                                           dtype=reused_cls.dtype)
                for c in range(num_classes):  # 循环处理 batch、FPN 层或类别维度，保持多尺度逻辑一致。
                    token_mask = class_pos_map[c] > 0.5
                    if token_mask.sum() > 0:  # 条件分支，根据当前状态选择不同处理路径。
                        tea_cls_6[:, :, c] = tea_cls[:, :, token_mask].max(dim=-1).values
                        reused_cls_6[:, :, c] = reused_cls[:, :, token_mask].max(dim=-1).values

                tea_soft = torch.sigmoid(tea_cls_6 / temperature).detach()  # [B, N, 6]

                # 只对 teacher 有响应的前景 anchor 计算
                fg_mask = tea_soft.max(dim=-1).values > 0.10  # [B, N]
                if fg_mask.sum() > 0:  # 条件分支，根据当前状态选择不同处理路径。
                    reused_sel = reused_cls_6[fg_mask]  # [num_fg, 6]
                    tea_sel = tea_soft[fg_mask]         # [num_fg, 6]
                    losses['loss_cls_kd'] = F.binary_cross_entropy_with_logits(  # 分类蒸馏损失，使学生增强分类响应逼近教师软标签。
                        reused_sel / temperature,
                        tea_sel,
                        reduction='mean'
                    ) * (temperature ** 2) * kd_weights.get('cls', 1.0)
                else:  # 兜底分支，处理上述条件都不满足的情况。
                    losses['loss_cls_kd'] = reused_cls.sum() * 0.0  # 分类蒸馏损失，使学生增强分类响应逼近教师软标签。
            else:  # 兜底分支，处理上述条件都不满足的情况。
                # positive_map 未就绪时的 fallback
                losses['loss_cls_kd'] = reused_cls.sum() * 0.0  # 分类蒸馏损失，使学生增强分类响应逼近教师软标签。

            # ============================================================
            # 2. 回归 KD（原逻辑保持不变）
            # ============================================================
            reg_loss = 0  # 累积回归蒸馏损失，让学生框预测逼近教师框预测。
            for t, s in zip(tea_bbox, reused_bbox):  # 循环处理 batch、FPN 层或类别维度，保持多尺度逻辑一致。
                if t.shape == s.shape:  # 条件分支，根据当前状态选择不同处理路径。
                    diff = torch.abs(t - s)
                    weight = torch.exp(diff)
                    reg_loss += (F.smooth_l1_loss(s, t, reduction='none') * weight).mean()
            if reg_loss > 0:  # 条件分支，根据当前状态选择不同处理路径。
                losses['loss_reg_kd'] = reg_loss / len(tea_bbox) * kd_weights.get('reg', 1.0)

            # ============================================================
            # 3. 中心度 KD（原逻辑保持不变）
            # ============================================================
            center_loss = 0  # 累积中心度蒸馏损失，让学生质量估计逼近教师中心度。
            for t, s in zip(tea_center, reused_center):  # 循环处理 batch、FPN 层或类别维度，保持多尺度逻辑一致。
                if t.shape == s.shape:  # 条件分支，根据当前状态选择不同处理路径。
                    center_loss += F.binary_cross_entropy_with_logits(
                        s, t.sigmoid(), reduction='mean')
            if center_loss > 0:  # 条件分支，根据当前状态选择不同处理路径。
                losses['loss_center_kd'] = center_loss / len(tea_center) * kd_weights.get('center', 1.0)

        except Exception as e:  # 捕获异常并给出安全回退，保证训练流程继续。
            print(f"Enhanced KD loss error: {e}")
            for key in ['loss_cls_kd', 'loss_reg_kd', 'loss_center_kd']:  # 循环处理 batch、FPN 层或类别维度，保持多尺度逻辑一致。
                if key not in losses:  # 条件分支，根据当前状态选择不同处理路径。
                    losses[key] = torch.tensor(0.0, device=device, requires_grad=False)

        return losses  # 返回训练框架需要反向传播的所有检测损失与 KD 损失。
    def pred_imitation_loss_single(self,  # 定义 pred_imitation_loss_single 函数/方法，承担当前蒸馏流程中的一个子步骤。
                                   labels,
                                   anchors,
                                   tea_cls_score,
                                   tea_bbox_pred,
                                   tea_centernesses,
                                   reused_cls_score,
                                   reused_bbox_pred,
                                   reused_centernesses,
                                   label_weights,
                                   bbox_targets,#add by jojo 20250425
                                   avg_factor):

        # classification branch distillation

        # print("tea_bbox_pred",tea_cls_score .shape)#[2,10000,256]

        # tea_cls_score = tea_cls_score.permute(0, 2, 3, 1).reshape(-1, self.bbox_head.cls_out_channels)
        # reused_cls_score = reused_cls_score.permute(0, 2, 3, 1).reshape(-1, self.bbox_head.cls_out_channels)
        # label_weights = label_weights.reshape(-1)

        #Add by jojo 20250423
        # tea_cls_score0 = tea_cls_score.reshape(-1, 256)  # [2,100000,256]
        # reused_cls_score0 = reused_cls_score.reshape(-1, 256)
        # label_weights0 = label_weights.reshape(-1)
        # Loss is not computed for the padded regions of the text.
        # ===== this change =====
        anchors = anchors.reshape(-1, 4)

        pos_inds = (labels.sum(-1) > 0).reshape(-1)  # 筛选正样本位置，回归和中心度 KD 只在正样本上更可靠。
        assert (self.text_masks.dim() == 2)  # 数值/结构检查，尽早发现 NaN、Inf 或维度不匹配问题。
        text_mask = (self.text_masks > 0).unsqueeze(1)#([2, 1, 256])
        text_mask = text_mask.repeat(1, tea_cls_score.size(1), 1)# torch.Size([2, 10000, 256])
        tea_cls_score= torch.masked_select(tea_cls_score, text_mask).contiguous()#focus on the valid area
        reused_cls_score=torch.masked_select(reused_cls_score, text_mask).contiguous()
        # print("tea_cls_score:",tea_cls_score)#logits value
        # labels1=labels.repeat(1, tea_cls_score.size(1), 1)
        # labels1 = torch.masked_select(labels1, text_mask)#error
        label_weights = label_weights[...,
        None].repeat(1, 1, text_mask.size(-1))
        label_weights = torch.masked_select(label_weights, text_mask)

        # print("reused_cls_score:",reused_cls_score.shape)#torch.Size([380000])
        # print("tea_cls_score",tea_cls_score.shape)#torch.Size([380000])
        # print("label_weights",label_weights.shape)#torch.Size([380000])

        tea_bbox_pred = tea_bbox_pred.reshape(-1, 4)
        reused_bbox_pred = reused_bbox_pred.reshape(-1, 4)

        tea_centernesses = tea_centernesses.reshape(-1)
        reused_centernesses = reused_centernesses.reshape(-1)

        label_weights = label_weights.reshape(-1)

        #------transformed into 2d
        tea_cls_score = tea_cls_score.repeat(1, 1).permute(1,0)
        reused_cls_score=reused_cls_score.repeat(1, 1).permute(1,0)
        #------end

        # -----add end-------
        # print("reused_cls_score:",reused_cls_score.shape)#torch.Size([507034])


        loss_cls_kd = self.loss_cls_kd(  # 调用配置的分类 KD 损失函数对齐教师与复用头分类输出。
            reused_cls_score,
            tea_cls_score,
            label_weights,
            avg_factor=avg_factor)

        # loss_cls_kd = self.loss_cls_kd(
        #     reused_cls_score,
        #     tea_cls_score)

        # print("loss_cls_kd:", loss_cls_kd)#yes!

        # regression branch distillation
        #add by jojo 20250423
        if pos_inds.sum() > 0:  # 条件分支，根据当前状态选择不同处理路径。
            #targets are teacher pos is student
            tea_bbox_pred= tea_bbox_pred[pos_inds]
            reused_bbox_pred = reused_bbox_pred[pos_inds]

            pos_anchors = anchors[pos_inds]
            pos_centerness = reused_centernesses[pos_inds]#centress reused

            centerness_targets = self.teacher.bbox_head.centerness_target(
                pos_anchors, tea_bbox_pred)#tea_centerness
            # print("tea_centernesses", tea_centernesses.shape)
            # print("centerness_targets",centerness_targets.shape)

            if torch.isnan(centerness_targets).any():  # 条件分支，根据当前状态选择不同处理路径。
                print('=====Centerness includes NaN=====')

                mask = ~torch.isnan(centerness_targets)
                # tea_centernesses = centerness_targets[mask]
                pos_centerness = pos_centerness[mask]
                pos_anchors = pos_anchors[mask]
                tea_bbox_pred = tea_bbox_pred[mask]
                reused_bbox_pred = reused_bbox_pred[mask]

                if tea_bbox_pred.shape[0] == 0:  # 条件分支，根据当前状态选择不同处理路径。
                    loss_bbox = reused_bbox_pred.sum() * 0
                    loss_centerness = pos_centerness.sum() * 0
                    # tea_centernesses= tea_bbox_pred .new_tensor(0.)
                    return loss_cls_kd, loss_bbox, loss_centerness  # 返回当前步骤的结果，供上层训练流程继续使用。

                    # The decoding process takes the offset into consideration.
            pos_anchors[:, 2:] += 1
            reused_bbox_pred = self.teacher.bbox_head.bbox_coder.decode(
                        pos_anchors, reused_bbox_pred)
            tea_bbox_pred=self.teacher.bbox_head.bbox_coder.decode(
                        pos_anchors, tea_bbox_pred)

            # print("reused_bbox_pred:",reused_bbox_pred.shape)
            # print(reused_bbox_pred)
            # print("tea_bbox_pred:",tea_bbox_pred.shape)
            # print(tea_bbox_pred)

            loss_reg_kd = self.loss_reg_kd(  # 调用配置的回归 KD 损失函数对齐解码后的预测框。
                reused_bbox_pred,
                tea_bbox_pred,
                weight=centerness_targets,
                avg_factor=1.0)
            # centerness loss
            # loss_center_kd  = self.loss_center_kd(
            #     pos_centerness, centerness_targets, avg_factor=avg_factor)
            loss_center_kd = self.loss_center_kd(  # 调用中心度 KD 损失函数对齐目标质量预测。
                pos_centerness, tea_centernesses[pos_inds].sigmoid(), avg_factor=avg_factor)
            # print("loss_reg_kd_0", loss_reg_kd)
            # print("loss_center_kd_0", loss_center_kd)
        else:  # 兜底分支，处理上述条件都不满足的情况。
            loss_reg_kd = reused_bbox_pred.sum() * 0  # 调用配置的回归 KD 损失函数对齐解码后的预测框。
            loss_center_kd = reused_centernesses.sum() * 0  # 调用中心度 KD 损失函数对齐目标质量预测。
            # print("loss_reg_kd_1",loss_reg_kd)
            # print("loss_center_kd_1",loss_center_kd)


        return loss_cls_kd, loss_reg_kd, loss_center_kd  # 返回当前步骤的结果，供上层训练流程继续使用。





class CrossModalAttention(nn.Module):  # 定义 CrossModalAttention 类，封装检测器或跨模态辅助模块。
    def __init__(self, v_dim, l_dim, num_heads=8):  # 定义 __init__ 函数/方法，承担当前蒸馏流程中的一个子步骤。
        super().__init__()  # 调用父类初始化，继承单阶段检测器和教师学生蒸馏基础能力。
        # 视觉特征投影
        self.vision_proj = nn.Sequential(  # 将视觉通道投影到与文本注意力一致的隐空间。
            nn.Conv2d(v_dim, l_dim, 1),
            nn.GroupNorm(8, l_dim)
        )

        # 文本特征压缩与投影
        self.text_compress = nn.Sequential(  # 压缩文本 token/位置维度，降低跨模态融合计算量。
            nn.Conv1d(256, 64, 3, padding=1),  # 压缩位置维度256→64
            nn.ReLU(inplace=True)
        )
        self.text_proj1 = nn.Linear(64, l_dim)  # 初始化/更新 self.text_proj1，作为模型状态或蒸馏模块配置。

        # 多头注意力机制
        self.attn = nn.MultiheadAttention(  # 定义多头注意力，用文本作为 key/value 引导视觉 query。
            embed_dim=l_dim,
            num_heads=num_heads,
            batch_first=True  # 使用batch_first格式
        )

    def forward(self, visual_feat, text_feat):  # 定义 forward 函数/方法，承担当前蒸馏流程中的一个子步骤。
        """
        visual_feat: [B, C, H, W]
        text_feat: [B, 256, 512] (D=256位置数，每个位置512维)
        """
        # 视觉特征处理
        B, C, H, W = visual_feat.size()
        v_proj = self.vision_proj(visual_feat)  # [B, l_dim, H, W]
        v_flat = v_proj.view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, l_dim]

        # 文本特征处理
        # 压缩位置维度 256→64
        text_compressed = self.text_compress(text_feat)  # [B, 64, 512] [2,64,512]
        # 投影到与视觉相同的维度空间
        l_proj = self.text_proj1(text_compressed.permute(0, 2, 1))  # [B, 512, 64] → [B, 64, l_dim]

        # 交叉注意力计算
        attn_out, _ = self.attn(  # 执行视觉到文本的交叉注意力，获得文本引导的视觉表示。
            query=v_flat,  # [B, HW, l_dim]
            key=l_proj,  # [B, 64, l_dim]
            value=l_proj  # [B, 64, l_dim]
        )

        # 恢复空间维度
        output = attn_out.permute(0, 2, 1).view(B, -1, H, W)
        return output  # 返回当前步骤的结果，供上层训练流程继续使用。


class TextAwareFPNLayer(nn.Module):  # 定义 TextAwareFPNLayer 类，封装检测器或跨模态辅助模块。
    def __init__(self, feat_dim):  # 定义 __init__ 函数/方法，承担当前蒸馏流程中的一个子步骤。
        super().__init__()  # 调用父类初始化，继承单阶段检测器和教师学生蒸馏基础能力。
        # 增加文本特征压缩层
        # 文本特征压缩层（处理3D文本特征）
        self.text_compress = nn.Sequential(  # 压缩文本 token/位置维度，降低跨模态融合计算量。
            nn.Conv1d(256, 64, 3, padding=1),  # 处理位置维度
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)  # 压缩位置维度 [B, 64, 1]
        )

        # 门控生成器
        self.text_gate = nn.Sequential(  # 初始化/更新 self.text_gate，作为模型状态或蒸馏模块配置。
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feat_dim),
            nn.Sigmoid()
        )

        # 特征融合层
        self.fusion = nn.Sequential(  # 初始化/更新 self.fusion，作为模型状态或蒸馏模块配置。
            nn.Conv2d(feat_dim * 2, feat_dim, 3, padding=1),
            nn.GroupNorm(8, feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, visual_feat, text_feat):  # 定义 forward 函数/方法，承担当前蒸馏流程中的一个子步骤。
        """
        visual_feat: [B, C, H, W] (C=feat_dim)
        text_feat: [B, 256, 512] (D=256位置数, 每个位置512维)
        """
        # 步骤1：文本特征压缩
        # [B, 256, 512] -> [B, 64, 512] -> [B, 64, 1]
        compressed_text = self.text_compress(text_feat)  # [B, 64, 1]

        # 步骤2：通道维度转换
        # [B, 64, 1] -> [B, 1, 512]
        text_feat = compressed_text.permute(0, 2, 1)  # [B, 1, 64]
        text_feat = F.interpolate(text_feat, size=256)  # [B, 1, 512]

        # 步骤3：生成门控信号
        # [B, 1, 512] -> [B, 512] -> [B, C]
        gate = self.text_gate(text_feat.squeeze(1))  # [B, C]
        gate = gate.view(-1, gate.size(1), 1, 1)  # [B, C, 1, 1]

        # 步骤4：门控增强
        gated_feat = visual_feat * gate  # [B, C, H, W]

        # 步骤5：残差融合
        fused = self.fusion(torch.cat([visual_feat, gated_feat], dim=1))  # 融合原始视觉特征和文本门控特征。
        return fused + visual_feat  # 返回当前步骤的结果，供上层训练流程继续使用。


class VisualEnhancedText(nn.Module):  # 定义 VisualEnhancedText 类，封装检测器或跨模态辅助模块。
    def __init__(self, text_dim=768, vis_dim=256, num_heads=8):  # 修改1：调整头数与维度
        super().__init__()  # 调用父类初始化，继承单阶段检测器和教师学生蒸馏基础能力。
        # 视觉上下文提取（适配768维度）
        self.vis_context = nn.Sequential(  # 初始化/更新 self.vis_context，作为模型状态或蒸馏模块配置。
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(vis_dim, text_dim, 1),  # 输出通道调整为768
            nn.LayerNorm([text_dim, 1, 1])
        )

        # 跨模态注意力（适配768维度）
        self.cross_attn = nn.MultiheadAttention(  # 初始化/更新 self.cross_attn，作为模型状态或蒸馏模块配置。
            embed_dim=text_dim,  # 768
            num_heads=num_heads,  # 头数12（768/64=12）
            batch_first=True,
            kdim=text_dim,  # 新增键值维度指定
            vdim=text_dim
        )

        # 门控融合（适配768维度）
        self.gate = nn.Sequential(  # 初始化/更新 self.gate，作为模型状态或蒸馏模块配置。
            nn.Linear(text_dim * 2, text_dim),  # 输入1536→768
            nn.Sigmoid()
        )

    def forward(self, text_feat, visual_feat):  # 定义 forward 函数/方法，承担当前蒸馏流程中的一个子步骤。
        """
        输入维度说明：
        text_feat:  [B, 256, 768] （L=256位置数，D=768维度）
        visual_feat: [B, 256, H, W]
        """
        # 视觉上下文提取
        vis_ctx = self.vis_context(visual_feat)  # [B,768,1,1]
        vis_ctx = vis_ctx.view(vis_ctx.size(0), -1).unsqueeze(1)  # [B,1,768]


        # print("text_feat:",text_feat.shape)#text_feat: torch.Size([2, 256, 768])
        # print("vis_ctx:",vis_ctx.shape)#vis_ctx: torch.Size([2, 1, 768])

        # 交叉注意力（Query=Text, Key/Value=Visual）
        attn_output, _ = self.cross_attn(
            query=text_feat,
            key=vis_ctx,
            value=vis_ctx
        )  # [B,768,256]
        # print("attn_output",attn_output.shape)

        # 残差连接
        # attn_output = attn_output.permute(0, 2, 1)  # [B,256,768]
        temp=torch.cat([text_feat, attn_output],dim=-1)
        # print("temp:", temp.shape) #temp: torch.Size([2, 256, 1536])
        gate = self.gate( temp)  # 根据文本语义生成通道门控权重。

        enhanced_text = gate * text_feat + (1 - gate) * attn_output  # 用门控在原文本和视觉注意力文本之间自适应融合。

        return enhanced_text  # 保持[B,256,768]

