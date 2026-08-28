# Copyright (c) FusionKD. All rights reserved.
from typing import Optional, Sequence

import numpy as np
from mmcv.transforms import to_tensor
from mmengine.structures import InstanceData

from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import BaseBoxes
from .formatting import PackDetInputs
from .loading import LoadAnnotations


@TRANSFORMS.register_module()
class LoadPseudoAnnotations(LoadAnnotations):
    """Load bbox/label annotations and keep pseudo scores.

    Extra added key:
        - gt_pseudo_scores (np.float32)
    """

    def _load_pseudo_scores(self, results: dict) -> None:
        gt_pseudo_scores = []
        for instance in results.get('instances', []):
            gt_pseudo_scores.append(float(instance.get('pseudo_score', 1.0)))
        results['gt_pseudo_scores'] = np.array(gt_pseudo_scores, dtype=np.float32)

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        self._load_pseudo_scores(results)
        return results


@TRANSFORMS.register_module()
class PackDetInputsWithPseudo(PackDetInputs):
    """PackDetInputs + gt_pseudo_scores -> gt_instances.pseudo_scores."""

    def transform(self, results: dict) -> dict:
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()
            packed_results['inputs'] = img

        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = to_tensor(results[key][valid_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(results[key])

        if 'gt_pseudo_scores' in results:
            if 'gt_ignore_flags' in results:
                instance_data.pseudo_scores = to_tensor(results['gt_pseudo_scores'][valid_idx])
                ignore_instance_data.pseudo_scores = to_tensor(results['gt_pseudo_scores'][ignore_idx])
            else:
                instance_data.pseudo_scores = to_tensor(results['gt_pseudo_scores'])

        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample
        return packed_results
