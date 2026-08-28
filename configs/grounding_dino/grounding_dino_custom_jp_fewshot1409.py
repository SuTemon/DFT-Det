_base_ = [
    './grounding_dino_swin-t_finetune_16xb2_1x_coco.py'
]

# data_root = '/media/csu/HDD/ZZW/IOD-complete/data/NEU-DET/star-1/coco/'
data_root = '/media/csu/HDD/szz/IOD-complete/data/1000_original_6class_0improve/y/'
metainfo = {
    # 'classes': ('huahen', 'damadian', 'bankong', 'liebian', 'huanzhangshamu', 'wuzhuangshamu', 'xiaoquanliebian', 'duorou', 'xiaomianbuliang'),
    # 'palette': [
    #     (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)
    # ]
        'classes': ('huahen', 'liebian', 'huanzhangshamu', 'wuzhuangshamu', 'xiaoquanliebian', 'xiaomianbuliang'),
        'palette':
            [
                (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100)
            ]
        # 'classes': ('crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches'),
        # 'palette':
        #     [
        #         (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100)
        #     ]
}

model = dict(
    bbox_head=dict(
        num_classes=6
    )
)
load_from = 'checkpoints/groundingdino_swint_ogc_mmdet-822d7e9d.pth'
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(960, 960), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(960, 960),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities'))
]
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        # ann_file='fewshot/fewshot_from_train/shots10/seed0/train.json',
        data_prefix=dict(img='train/images/'),
        pipeline=train_pipeline,
        return_classes=True
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='val/images/'),
        pipeline=test_pipeline,
        return_classes=True
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    format_only=False,
    classwise=True
)
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)

auto_scale_lr = dict(base_batch_size=12)