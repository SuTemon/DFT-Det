# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import cat_boxes
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from ..utils import images_to_levels, multi_apply, unpack_gt_instances
from .single_stage import SingleStageDetector


@MODELS.register_module()
class FusionKDSingleStageDetector(SingleStageDetector):
    r"""Implementation of `Distilling the Knowledge in a Neural Network.
    <https://arxiv.org/abs/1503.02531>`_.

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone module.
        neck (:obj:`ConfigDict` or dict): The neck module.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head module.
        teacher_config (:obj:`ConfigDict` | dict | str | Path): Config file
            path or the config object of teacher model.
        teacher_ckpt (str, optional): Checkpoint path of teacher model.
            If left as None, the model will not load any weights.
            Defaults to True.
        eval_teacher (bool): Set the train mode for teacher.
            Defaults to True.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of ATSS. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of ATSS. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
    """

    def __init__(
        self,
        backbone: ConfigType,
        neck: ConfigType,
        bbox_head: ConfigType,
        teacher_config: Union[ConfigType, str, Path],
        teacher_ckpt: Optional[str] = None,
        teacher_freeze_cfg: Optional[dict] = None,
        kd_cfg: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
    ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor)
        # Build teacher model
        self.teacher_freeze_cfg = teacher_freeze_cfg or dict(freeze_all=True)
        if isinstance(teacher_config, (str, Path)):
            teacher_config = Config.fromfile(teacher_config)
        self.teacher = MODELS.build(teacher_config['model'])
        if teacher_ckpt is not None:
            load_checkpoint(self.teacher, teacher_ckpt, map_location='cpu')
        # In order to reforward teacher model, configurable freeze teacher.
        self.apply_teacher_freeze_cfg(self.teacher, self.teacher_freeze_cfg)
        self.loss_cls_kd = MODELS.build(kd_cfg['loss_cls_kd'])
        self.loss_reg_kd = MODELS.build(kd_cfg['loss_reg_kd'])
        self.with_feat_distill = False
        if kd_cfg.get('loss_feat_kd', None):
            self.loss_feat_kd = MODELS.build(kd_cfg['loss_feat_kd'])
            self.with_feat_distill = True
        self.reused_teacher_head_idx = kd_cfg['reused_teacher_head_idx']

    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def apply_teacher_freeze_cfg(self, teacher: nn.Module, freeze_cfg: dict) -> None:
        """Apply freeze configuration to teacher model.
        
        Args:
            teacher: The teacher model to freeze.
            freeze_cfg: Configuration dict with keys:
                - freeze_all (bool): If True, freeze all parameters.
                - frozen_modules (list): List of module names to freeze.
                - trainable_modules (list): List of module names to keep trainable.
        """
        freeze_all = freeze_cfg.get('freeze_all', True)
        frozen_modules = freeze_cfg.get('frozen_modules', []) or []
        trainable_modules = freeze_cfg.get('trainable_modules', []) or []
        
        if freeze_all:
            # Freeze all parameters first
            for param in teacher.parameters():
                param.requires_grad = False
            
            # Then unfreeze specified trainable modules
            if trainable_modules:
                for name, module in teacher.named_modules():
                    if any(tm in name for tm in trainable_modules):
                        for param in module.parameters():
                            param.requires_grad = True
        else:
            # Keep all trainable by default
            for param in teacher.parameters():
                param.requires_grad = True
            
            # Then freeze specified frozen modules
            if frozen_modules:
                for name, module in teacher.named_modules():
                    if any(fm in name for fm in frozen_modules):
                        for param in module.parameters():
                            param.requires_grad = False
        
        # Set teacher to eval mode
        teacher.eval()

    def _set_teacher_frozen_modules_eval(self) -> None:
        """Set frozen modules in teacher to eval mode."""
        frozen_modules = self.teacher_freeze_cfg.get('frozen_modules', []) or []
        if frozen_modules:
            for name, module in self.teacher.named_modules():
                if any(fm in name for fm in frozen_modules):
                    module.eval()

    def cuda(self, device: Optional[str] = None) -> nn.Module:
        """Since teacher is registered as a plain object, it is necessary to
        put the teacher model to cuda when calling ``cuda`` function."""
        self.teacher.cuda(device=device)
        return super().cuda(device=device)

    def to(self, device: Optional[str] = None) -> nn.Module:
        """Since teacher is registered as a plain object, it is necessary to
        put the teacher model to other device when calling ``to`` function."""
        self.teacher.to(device=device)
        return super().to(device=device)

    def train(self, mode: bool = True) -> None:
        """Set train mode while keeping configured frozen teacher parts in eval."""
        super().train(mode)
        if getattr(self, 'teacher_freeze_cfg', dict(freeze_all=True)).get('freeze_all', True):
            trainable_modules = self.teacher_freeze_cfg.get('trainable_modules', []) or []
            if not trainable_modules:
                self.teacher.train(False)
            else:
                self.teacher.train(mode)
                self._set_teacher_frozen_modules_eval()
        else:
            self.teacher.train(mode)
            self._set_teacher_frozen_modules_eval()

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        这段代码的含义是：self.teacher = ... 时不会走 PyTorch 的 nn.Module.__setattr__ 注册逻辑，而是直接作为普通属性挂上去。
        因此：
        self.parameters() 不会包含 teacher 参数；
        optimizer 构建时通常只拿 model.parameters()，所以 teacher 参数不会进入 optimizer；
        即便某些参数忘记 requires_grad=False，它们也不会被优化器更新。
        所以这里是“双保险”：
        teacher 参数 requires_grad=False；
        teacher 不注册为主模型子模块，不进入 optimizer。
        """
        if name == 'teacher':
            teacher_cfg = getattr(self, 'teacher_freeze_cfg', dict(freeze_all=True))
            freeze_all = teacher_cfg.get('freeze_all', True)
            trainable_modules = teacher_cfg.get('trainable_modules', []) or []
            default_register = (not freeze_all) or bool(trainable_modules)
            register_teacher = teacher_cfg.get('register_teacher', default_register)
            if register_teacher:
                super().__setattr__(name, value)
            else:
                object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)
