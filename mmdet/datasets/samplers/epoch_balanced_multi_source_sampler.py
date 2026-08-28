# Copyright (c) OpenMMLab. All rights reserved.
import math
import random
from typing import Iterator, List

import torch
import torch.distributed as dist

from mmdet.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class EpochBalancedMultiSourceSampler:
    """适配 EpochBasedTrainLoop 的有限长度多源采样器。

    适用场景：
        ConcatDataset([fewshot_dataset, pseudo_dataset])

    功能：
        1. 每个 batch 按 source_ratio 从多个子数据集取样；
        2. 每个 epoch 只产出固定数量样本，保证 EpochBasedTrainLoop 正常结束；
        3. pseudo 数据不需要物理截断，每个 epoch 从完整 pseudo 池中随机采样；
        4. 支持 DDP 分布式训练；
        5. 支持 set_epoch，使每个 epoch 的采样顺序不同。

    推荐配置：
        batch_size=2
        source_ratio=[1, 1]
        即每个 batch = 1 few-shot + 1 pseudo
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        source_ratio: List[int],
        shuffle: bool = True,
        seed: int = 0,
        round_up: bool = False,
        epoch_size_source: int = 0,
    ):
        assert hasattr(dataset, 'datasets'), (
            'EpochBalancedMultiSourceSampler expects a ConcatDataset, '
            'but got dataset without attribute "datasets".'
        )

        assert len(dataset.datasets) == len(source_ratio), (
            f'len(dataset.datasets)={len(dataset.datasets)} must equal '
            f'len(source_ratio)={len(source_ratio)}.'
        )

        assert batch_size == sum(source_ratio), (
            f'For this sampler, batch_size should equal sum(source_ratio). '
            f'Got batch_size={batch_size}, sum(source_ratio)={sum(source_ratio)}.'
        )

        self.dataset = dataset
        self.datasets = dataset.datasets
        self.batch_size = batch_size
        self.source_ratio = list(source_ratio)
        self.shuffle = shuffle
        self.seed = seed
        self.round_up = round_up
        self.epoch = 0

        # 用哪个 source 决定 epoch 长度。
        # 默认 0，即 fewshot_dataset。
        self.epoch_size_source = epoch_size_source

        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        # ConcatDataset 的全局 index 偏移。
        # 例如 len(fewshot)=63，则：
        #   fewshot index: 0~62
        #   pseudo  index: 63~63+len(pseudo)-1
        if hasattr(dataset, 'cumulative_sizes'):
            cumulative_sizes = list(dataset.cumulative_sizes)
        else:
            cumulative_sizes = []
            running = 0
            for ds in self.datasets:
                running += len(ds)
                cumulative_sizes.append(running)

        self.source_offsets = [0] + cumulative_sizes[:-1]

        # 每个 batch 从每个 source 取多少张
        self.samples_per_source = list(source_ratio)

        # 关键：由 few-shot source 决定每个 epoch 的 batch 数。
        # source_ratio=[1,1] 时：
        #   num_batches = len(fewshot_dataset) // 1
        ref_len = len(self.datasets[self.epoch_size_source])
        ref_per_batch = self.samples_per_source[self.epoch_size_source]

        self.num_batches = ref_len // ref_per_batch

        if self.num_batches <= 0:
            raise ValueError(
                f'num_batches={self.num_batches}. '
                f'Please check dataset length={ref_len} and '
                f'ref_per_batch={ref_per_batch}.'
            )

        # 分布式情况下，每张卡拿一部分 batch。
        if self.round_up:
            self.num_batches_per_rank = math.ceil(
                self.num_batches / self.world_size)
        else:
            self.num_batches_per_rank = self.num_batches // self.world_size

        if self.num_batches_per_rank <= 0:
            raise ValueError(
                f'num_batches_per_rank={self.num_batches_per_rank}. '
                f'num_batches={self.num_batches}, world_size={self.world_size}. '
                f'Few-shot samples may be too few for current GPU number.'
            )

        self.total_batches = self.num_batches_per_rank * self.world_size
        self.num_samples = self.num_batches_per_rank * self.batch_size

    def __len__(self) -> int:
        """返回当前 rank 一个 epoch 内产出的样本数。

        DataLoader 的 len 通常约等于：
            len(sampler) // batch_size

        因此 batch_size=2、fewshot=63 时：
            len(sampler)=126
            len(dataloader)=63
        """
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _build_source_indices(self, source_id: int, need: int) -> List[int]:
        """为某个 source 构建本 epoch 需要的全局 index。"""
        dataset_len = len(self.datasets[source_id])
        local_indices = list(range(dataset_len))

        # 每个 source、每个 epoch 使用不同随机种子
        rng = random.Random(self.seed + self.epoch * 10007 + source_id * 97)

        if self.shuffle:
            rng.shuffle(local_indices)

        if dataset_len >= need:
            selected = local_indices[:need]
        else:
            # few-shot 极少时允许重复采样补齐
            repeat = math.ceil(need / dataset_len)
            selected = (local_indices * repeat)[:need]

        offset = self.source_offsets[source_id]
        global_indices = [idx + offset for idx in selected]
        return global_indices

    def __iter__(self) -> Iterator[int]:
        # 每个 source 在当前 epoch 需要多少样本
        # 例如 total_batches=63, source_ratio=[1,1]：
        #   fewshot need = 63
        #   pseudo  need = 63
        if self.rank == 0:
            print(
                f'[EpochBalancedMultiSourceSampler] '
                f'epoch={self.epoch}, '
                f'num_batches={self.num_batches_per_rank}, '
                f'batch_size={self.batch_size}, '
                f'source_ratio={self.source_ratio}, '
                f'len_source0={len(self.datasets[0])}, '
                f'len_source1={len(self.datasets[1])}'
            )
        source_indices = []
        for source_id, num_per_batch in enumerate(self.samples_per_source):
            need = self.total_batches * num_per_batch
            indices = self._build_source_indices(source_id, need)
            source_indices.append(indices)

        ptrs = [0 for _ in self.datasets]
        all_batches = []

        batch_rng = random.Random(self.seed + self.epoch * 10009)

        for _ in range(self.total_batches):
            batch = []
            for source_id, num_per_batch in enumerate(self.samples_per_source):
                start = ptrs[source_id]
                end = start + num_per_batch
                batch.extend(source_indices[source_id][start:end])
                ptrs[source_id] = end

            # batch 内顺序可以打乱，避免永远 few-shot 在前、pseudo 在后
            if self.shuffle:
                batch_rng.shuffle(batch)

            all_batches.append(batch)

        # DDP：不同 rank 拿不同 batch
        rank_batches = all_batches[self.rank:self.total_batches:self.world_size]

        # 注意：这里作为 sampler 使用，需要 yield 单个 index，
        # DataLoader 会再根据 batch_size=2 自动组 batch。
        flatten_indices = []
        for batch in rank_batches:
            flatten_indices.extend(batch)

        return iter(flatten_indices)