# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import tempfile
import time
from pathlib import Path

import torch
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

from mmdet.registry import MODELS


def build_grounding_dino_forward_wrapper(detector, data_samples):
    text_prompts = []
    for sample in data_samples:
        custom_entities = sample.get('custom_entities', False)
        _, caption, _, _ = detector.get_tokens_positive_and_prompts(
            sample.text, custom_entities)
        text_prompts.append(caption)

    class GroundingDINOForwardWrapper(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.detector = detector

        def forward(self, inputs):
            text_dict = self.detector.language_model(text_prompts)
            text_dict['embedded'] = self.detector.text_feat_map(
                text_dict['embedded'])
            img_feats = self.detector.extract_feat(inputs)
            head_inputs = self.detector.forward_transformer(
                img_feats, text_dict, data_samples)
            return self.detector.bbox_head.forward(**head_inputs)

    return GroundingDINOForwardWrapper()


def build_glip_forward_wrapper(detector, data_samples):
    text_prompts = []
    for sample in data_samples:
        custom_entities = sample.get('custom_entities', False)
        _, caption, _, _ = detector.get_tokens_and_prompts(
            sample.text, custom_entities)
        text_prompts.append(caption)

    class GLIPForwardWrapper(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.detector = detector

        def forward(self, inputs):
            language_dict_features = self.detector.language_model(
                text_prompts)
            visual_features = self.detector.extract_feat(inputs)
            return self.detector.bbox_head.forward(visual_features,
                                                   language_dict_features)

    return GLIPForwardWrapper()


try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')


def parse_args():
    parser = argparse.ArgumentParser(description='Get a detector flops')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 800],
        help='input image size')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--timing-mode',
        choices=['forward', 'e2e'],
        default='forward',
        help='timing mode: forward measures model forward only; e2e measures '
        'data preprocessing plus model forward')
    args = parser.parse_args()
    return args


def measure_inference_time(model, data_batch, timing_mode, repeat_times=50,
                           warmup_iters=10):
    infer_times = []

    if timing_mode == 'forward':
        if hasattr(model, 'language_model'):
            run_model = build_grounding_dino_forward_wrapper(
                model, data_batch['data_samples'])
        else:
            run_model = model
        inputs = data_batch['inputs']

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for _ in range(warmup_iters):
            with torch.no_grad():
                _ = run_model(inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with torch.no_grad():
            for _ in range(repeat_times):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                _ = run_model(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                infer_times.append((time.perf_counter() - start_time) * 1000)

    elif timing_mode == 'e2e':
        raw_data_batch = {
            'inputs': [sample.cpu() for sample in data_batch['inputs']],
            'data_samples': data_batch['data_samples']
        }

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for _ in range(warmup_iters):
            with torch.no_grad():
                processed = model.data_preprocessor(raw_data_batch)
                if hasattr(model, 'language_model'):
                    _ = build_grounding_dino_forward_wrapper(
                        model, processed['data_samples'])(processed['inputs'])
                else:
                    _ = model(processed['inputs'])
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with torch.no_grad():
            for _ in range(repeat_times):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                processed = model.data_preprocessor(raw_data_batch)
                if hasattr(model, 'language_model'):
                    _ = build_grounding_dino_forward_wrapper(
                        model, processed['data_samples'])(processed['inputs'])
                else:
                    _ = model(processed['inputs'])
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                infer_times.append((time.perf_counter() - start_time) * 1000)
    else:
        raise ValueError(f'Unsupported timing mode: {timing_mode}')

    avg_infer_time_ms = sum(infer_times) / len(infer_times)
    fps = 1000.0 / avg_infer_time_ms
    return avg_infer_time_ms, fps


def inference(args, logger):
    if str(torch.__version__) < '1.12':
        logger.warning(
            'Some config files, such as configs/yolact and configs/detectors,'
            'may have compatibility issues with torch.jit when torch<1.12. '
            'If you want to calculate flops for these models, '
            'please make sure your pytorch version is >=1.12.')

    config_name = Path(args.config)
    if not config_name.exists():
        logger.error(f'{config_name} not found.')

    cfg = Config.fromfile(args.config)
    cfg.work_dir = tempfile.TemporaryDirectory().name
    cfg.log_level = 'WARN'
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    # TODO: The following usage is temporary and not safe
    # use hard code to convert mmSyncBN to SyncBN. This is a known
    # bug in mmengine, mmSyncBN requires a distributed environment，
    # this question involves models like configs/strong_baselines
    if hasattr(cfg, 'head_norm_cfg'):
        cfg['head_norm_cfg'] = dict(type='SyncBN', requires_grad=True)
        cfg['model']['roi_head']['bbox_head']['norm_cfg'] = dict(
            type='SyncBN', requires_grad=True)
        cfg['model']['roi_head']['mask_head']['norm_cfg'] = dict(
            type='SyncBN', requires_grad=True)

    if len(args.shape) == 1:
        h = w = args.shape[0]
    elif len(args.shape) == 2:
        h, w = args.shape
    else:
        raise ValueError('invalid input shape')
    result = {}
    infer_times = []

    # Supports two ways to calculate flops,
    # 1. randomly generate a picture
    # 2. load a picture from the dataset
    # In two stage detectors, _forward need batch_samples to get
    # rpn_results_list, then use rpn_results_list to compute flops,
    # so only the second way is supported
    try:
        model = MODELS.build(copy.deepcopy(cfg.model))
        if torch.cuda.is_available():
            model.cuda()
        model = revert_sync_batchnorm(model)
        data_batch = {'inputs': [torch.rand(3, h, w)], 'batch_samples': [None]}
        data = model.data_preprocessor(data_batch)
        result['ori_shape'] = (h, w)
        result['pad_shape'] = data['inputs'].shape[-2:]
        model.eval()
        outputs = get_model_complexity_info(
            model,
            None,
            inputs=data['inputs'],
            show_table=False,
            show_arch=False)
        flops = outputs['flops']
        params = outputs['params']
        result['compute_type'] = 'direct: randomly generate a picture'

    except (TypeError, AssertionError):
        logger.warning(
            'Failed to directly get FLOPs, try to get flops with real data')
        data_loader = Runner.build_dataloader(cfg.val_dataloader)
        data_batch = next(iter(data_loader))
        model = MODELS.build(copy.deepcopy(cfg.model))
        if torch.cuda.is_available():
            model = model.cuda()
        model = revert_sync_batchnorm(model)
        model.eval()
        data = model.data_preprocessor(data_batch)
        result['ori_shape'] = data['data_samples'][0].ori_shape
        result['pad_shape'] = data['data_samples'][0].pad_shape

        del data_loader
        if hasattr(model, 'language_model'):
            model_name = model.__class__.__name__.lower()
            if 'grounding' in model_name:
                model = build_grounding_dino_forward_wrapper(
                    model, data['data_samples'])
            elif 'glip' in model_name:
                model = build_glip_forward_wrapper(model, data['data_samples'])
            elif hasattr(model, 'text_feat_map'):
                model = build_grounding_dino_forward_wrapper(
                    model, data['data_samples'])
            else:
                model = build_glip_forward_wrapper(model, data['data_samples'])

        outputs = get_model_complexity_info(
            model,
            None,
            inputs=data['inputs'],
            show_table=False,
            show_arch=False)
        flops = outputs['flops']
        params = outputs['params']
        result['compute_type'] = 'dataloader: load a picture from the dataset'
        avg_infer_time_ms, fps = measure_inference_time(
            model, data, args.timing_mode)
        result['avg_infer_time_ms'] = avg_infer_time_ms
        result['fps'] = fps

    flops = _format_size(flops)
    params = _format_size(params)
    result['flops'] = flops
    result['params'] = params

    return result


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')
    result = inference(args, logger)
    split_line = '=' * 30
    ori_shape = result['ori_shape']
    pad_shape = result['pad_shape']
    flops = result['flops']
    params = result['params']
    compute_type = result['compute_type']
    avg_infer_time_ms = result.get('avg_infer_time_ms')
    fps = result.get('fps')

    if pad_shape != ori_shape:
        print(f'{split_line}\nUse size divisor set input shape '
              f'from {ori_shape} to {pad_shape}')
    extra_lines = ''
    if avg_infer_time_ms is not None and fps is not None:
        extra_lines = (f'\nTiming mode: {args.timing_mode}'
                       f'\nAvg infer time: {avg_infer_time_ms:.3f} ms'
                       f'\nFPS: {fps:.2f}')
    print(f'{split_line}\nCompute type: {compute_type}\n'
          f'Input shape: {pad_shape}\nFlops: {flops}\n'
          f'Params: {params}{extra_lines}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify '
          'that the flops computation is correct.')


if __name__ == '__main__':
    main()
