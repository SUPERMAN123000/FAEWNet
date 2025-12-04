_base_ = [
    '../_base_/models/ban_vit-b16.py', 
    '../common/standard_512x512_40k_levircd.py']

crop_size = (512, 512)

train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotFlip', rotate_prob=0.5, flip_prob=0.5, degree=(-20, 20)),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgExchangeTime', prob=0.5),
    dict(
        type='MultiImgPhotoMetricDistortion',
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10),
    dict(type='MultiImgPackSegInputs')
]

model = dict(
    asymetric_input=True,
    encoder_resolution=dict(
        size=(224, 224),
        mode='bilinear'),
    image_encoder=dict(
        frozen_exclude=[]),
    decode_head=dict(
        type='BitemporalAdapterHead',
        ban_cfg=dict(
            clip_channels=768,
            fusion_index=[0, 1, 2],
            side_enc_cfg=dict(
                type='mmseg.ResNetV1c',
                init_cfg=dict(
                    type='Pretrained', checkpoint='open-mmlab://resnet18_v1c'),
                in_channels=3,
                depth=18,
                num_stages=3,
                out_indices=(2,),
                dilations=(1, 1, 1),
                strides=(1, 2, 1),
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True)),
        ban_dec_cfg=dict(
            type='BAN_BITHead',
            in_channels=256,
            channels=32,
            num_classes=2)),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0]//2, crop_size[1]//2)))

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'img_encoder': dict(lr_mult=0.1, decay_mult=1.0),
            'norm': dict(decay_mult=0.),
            'mask_decoder': dict(lr_mult=10.)
        }),
    loss_scale='dynamic',
    clip_grad=dict(max_norm=0.01, norm_type=2))

train_dataloader = dict(batch_size=8, num_workers=8,dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=1, num_workers=1)