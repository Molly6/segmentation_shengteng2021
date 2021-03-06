# model settings
norm_cfg = dict(type='BN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=False)

model = dict(
    type='EncoderDecoder',
    pretrained='pretrain_new/swin_base_patch4_window12_384_22k.pth',
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=128,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        frozen_stages=2,
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg,
        ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=384,
        dropout_ratio=0.1,
        num_classes=47,
        norm_cfg=norm_cfg,
        align_corners=False,
        # loss_decode=dict(
        #     type='SoftCrossEntropyLoss', smooth_factor=0.1, ignore_index=255, loss_weight=1.0),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)
            ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=47,
        norm_cfg=norm_cfg,
        align_corners=False,
        # loss_decode=dict(
        #     type='SoftCrossEntropyLoss', smooth_factor=0.1, ignore_index=255, loss_weight=0.4)
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
            ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

albu_train_transforms = [
    dict(type='Flip', p=0.5),
]

# dataset settings
dataset_type = 'PascalVOCDataset' 
data_root = '/cache/dataset/' 
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_scale = (512, 512)
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile', channel_type='ngb'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=crop_size, ratio_range=(0.75, 1.25)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.0),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        update_pad_shape=False
        ),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', channel_type='ngb'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        img_ratios=[1.25], # [0.75, 1.0, 1.25],
        flip=False,
        flip_scale=[False, False, False],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=16, # 16
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        img_dir='/train/images',
        ann_dir='/train/labels',
        split='/workspace/mmsegmentation-master/trainval.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_dir='/train/images',
        ann_dir='/train/labels',
        split='/workspace/mmsegmentation-master/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_dir='/train/images',
        ann_dir='/train/labels',
        split='/workspace/mmsegmentation-master/val.txt',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(
    # _delete_=True,
    type='AdamW',
    lr=0.00006, 
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
        }))
optimizer_config = dict()
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512., distributed=False)

# fp16 placeholder
fp16 = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=3)
evaluation = dict(interval=80000, metric='mIoU', pre_eval=True) 
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

# custom_hooks = [dict(
#                 type='LinearMomentumEMAHook',
#                 momentum=0.001,
#                 interval=1,
#                 warm_up=100,
#                 resume_from=None
#                 )]
