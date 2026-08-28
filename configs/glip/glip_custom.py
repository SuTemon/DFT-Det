_base_ = [
    './glip_atss_swin-t_b_fpn_dyhead_16xb2_ms-2x_funtune_coco.py'
]

# data_root = '/media/csu/HDD/szz/IOD-complete/data/NEU-DET/star-1/coco/'
data_root = '/media/csu/data/szz/DFKD/datasets/NEU-DET/star-1/coco/'
metainfo = {
    # 'classes': ('huahen', 'damadian', 'bankong', 'liebian', 'huanzhangshamu', 'wuzhuangshamu', 'xiaoquanliebian', 'duorou', 'xiaomianbuliang'),
    # 'palette': [
    #     (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)
    # ]
            'classes': ('crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches'),
        'palette':
            [
                (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100)
            ]
}
model = dict(
    # Freeze control examples:
    # 1) 全部可训练: freeze_cfg=dict(freeze_all=False)
    # 2) 全部冻结: freeze_cfg=dict(freeze_all=True)
    # 3) 只冻结部分模块: freeze_cfg=dict(frozen_modules=['language_model', 'backbone'])
    # 4) 全部冻结但只训练检测头: freeze_cfg=dict(freeze_all=True, trainable_modules=['bbox_head'])
    freeze_cfg=dict(freeze_all=False),
    bbox_head=dict(
        num_classes=6,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
    )
)
load_from = 'checkpoints/glip_tiny_b_mmdet-6dfbd102.pth'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        imdecode_backend='pillow',
        backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 
                   'text', 'custom_entities') 
    )
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(640, 640),
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
        # ann_file='annotations/instances_train.json',
        ann_file='fewshot/fewshot_from_train/shots10/seed0/train.json',
        # data_prefix=dict(img='images/train2017/'),
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
        return_classes=True
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val2017.json',
        # data_prefix=dict(img='images/val2017/'),
        data_prefix=dict(img='val2017/'),
        pipeline=test_pipeline,
        return_classes=True
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
    classwise=True
)
test_evaluator = val_evaluator

test_dataloader = val_dataloader

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW',
#         lr=0.00001,
#         weight_decay=0.05
#     )
# )

auto_scale_lr = dict(base_batch_size=2)