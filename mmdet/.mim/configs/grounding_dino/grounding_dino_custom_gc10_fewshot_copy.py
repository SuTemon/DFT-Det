_base_ = [
    './grounding_dino_swin-t_finetune_16xb2_1x_coco.py'
]

# data_root = '/media/csu/HDD/szz/IOD-complete/data/NEU-DET/star-1/coco/'
# data_root = '/media/csu/ZZW/IOD-complete/data/NEU-DET/star-1/coco/'
data_root = '/media/csu/data/szz/IOD-complete/data/gc10/coco/'
# data_root = '/media/csu/HDD/ZZW/IOD-complete/data/1000_original_6class_0improve/y/'
metainfo = {
    # 'classes': ('huahen', 'damadian', 'bankong', 'liebian', 'huanzhangshamu', 'wuzhuangshamu', 'xiaoquanliebian', 'duorou', 'xiaomianbuliang'),
    # 'palette': [
    #     (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)
    # ]
        # 'classes': ('huahen', 'liebian', 'huanzhangshamu', 'wuzhuangshamu', 'xiaoquanliebian', 'xiaomianbuliang'),
        # 'palette':
        #     [
        #         (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100)
        #     ]
        'classes': ('punching_hole','welding_line','crescent_gap','water_spot','oil_spot','silk_spot','inclusion','rolled_pit','crease','waist folding'),
        'palette':
            [
                    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
            ]
        # 'classes': ('crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches'),
        # 'palette':
        #     [
        #         (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100)
        #     ]
}

model = dict(
    # Freeze control examples:
    # 1) 全部可训练: freeze_cfg=dict(freeze_all=False)
    # 2) 全部冻结: freeze_cfg=dict(freeze_all=True)
    # 3) 只冻结部分模块: freeze_cfg=dict(frozen_modules=['language_model', 'backbone'])
    # 4) 全部冻结但只训练检测头: freeze_cfg=dict(freeze_all=True, trainable_modules=['bbox_head'])
    freeze_cfg=dict(freeze_all=False),
    bbox_head=dict(
        num_classes=6
    )
)
load_from = 'checkpoints/groundingdino_swint_ogc_mmdet-822d7e9d.pth'
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', scale=(960, 960), keep_ratio=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
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
        # scale=(960, 960),
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
        # ann_file='fewshot/fewshot_from_train/shots10/seed3/train.json',
        ann_file='fewshot//shots10/seed3/train.json',
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
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
    classwise=True
)
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)

auto_scale_lr = dict(base_batch_size=12)