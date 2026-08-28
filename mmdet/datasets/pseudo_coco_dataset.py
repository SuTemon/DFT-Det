# Copyright (c) FusionKD. All rights reserved.
"""
PseudoCocoDataset
=================
继承自 CocoDataset，专门解析 filter_pseudo_2.py 生成的 pseudo_coco.json。

关键扩展
--------
1. 每个 instance 额外保留 ``pseudo_score`` (quality_score)，
   在打包时写入 ``gt_instances.pseudo_scores``，供 model 层按质量加权 loss。
2. 提供 ``pseudo_score_floor`` 参数，在 load 阶段过滤掉过低质量伪标签。
3. 数据集标识字段 ``is_pseudo = True``，供 loss 函数区分两路 gt。

使用方式（config 示例）
-----------------------
pseudo_dataset = dict(
    type='PseudoCocoDataset',
    ann_file='/path/to/pseudo_coco.json',
    data_prefix=dict(img='/path/to/train/images/'),
    metainfo=dict(classes=('crazing', ...), palette=[...]),
    pseudo_score_floor=0.30,   # 低于该值的伪标签丢弃
    pipeline=[...],
)
"""

import copy
import os.path as osp
from typing import List, Union

from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class PseudoCocoDataset(CocoDataset):
    """从 pseudo_coco.json 加载伪标签，附带 quality_score 信息。

    Args:
        pseudo_score_floor (float): 低于该阈值的 pseudo annotation 在 load
            阶段直接丢弃，默认 0.0（不过滤，由 filter_pseudo_2.py 已过滤）。
        pseudo_score_key (str): pseudo_coco.json 中存储质量分数的字段名，
            默认 ``'pseudo_score'``（filter_pseudo_2.py 的输出字段）。
    """

    def __init__(
        self,
        pseudo_score_floor: float = 0.0,
        pseudo_score_key: str = 'pseudo_score',
        **kwargs,
    ):
        self.pseudo_score_floor = pseudo_score_floor
        self.pseudo_score_key   = pseudo_score_key
        super().__init__(**kwargs)

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """在父类解析基础上，为每个 instance 附加 pseudo_score。"""
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix,
            )
        else:
            seg_map_path = None

        data_info['img_path']     = img_path
        data_info['img_id']       = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height']       = img_info['height']
        data_info['width']        = img_info['width']
        data_info['is_pseudo']    = True   # 标记该数据来自伪标签

        if self.return_classes:
            data_info['text']           = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for ann in ann_info:
            instance = {}

            if ann.get('ignore', False):
                continue

            # 读取质量分数，若字段缺失则赋予默认值 1.0
            pseudo_score = float(ann.get(self.pseudo_score_key, 1.0))
            if pseudo_score < self.pseudo_score_floor:
                continue

            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width'])  - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue

            bbox = [x1, y1, x1 + w, y1 + h]

            instance['ignore_flag']   = 1 if ann.get('iscrowd', False) else 0
            instance['bbox']          = bbox
            instance['bbox_label']    = self.cat2label[ann['category_id']]
            instance['pseudo_score']  = pseudo_score  # ← 质量分数

            # 额外保存子分量，供 loss 层调试/分析
            instance['pseudo_mean_score'] = float(ann.get('pseudo_mean_score', pseudo_score))
            instance['pseudo_tta_score']  = float(ann.get('tta_score',         1.0))
            instance['pseudo_geom_score'] = float(ann.get('geometry_score',    1.0))

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)

        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """过滤空标注图像（伪标签场景中过滤掉全部被丢弃的图）。"""
        if self.test_mode:
            return self.data_list

        valid = []
        for data_info in self.data_list:
            if len(data_info.get('instances', [])) > 0:
                valid.append(data_info)
        return valid
