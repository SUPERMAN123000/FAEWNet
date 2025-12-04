_base_ = ['./ban_vit-b16-clip_bit_512x512_80k_s2looking.py']

pretrained = 'pretrain/clip_vit-large-patch14-336_3rdparty-0b5df9cb.pth'  # noqa

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
    type='BAN',
    pretrained=pretrained,
    encoder_resolution=dict(
        size=(336, 336),
        mode='bilinear'),
    image_encoder=dict(
        type='mmseg.VisionTransformer',
        img_size=(336, 336),
        patch_size=14,
        patch_pad=0,
        embed_dims=1024,
        num_layers=18,
        num_heads=16,
        out_indices=(5, 11, 17)),
    decode_head=dict(
        type='BitemporalAdapterHead',
        ban_cfg=dict(
            fusion_index=[0, 1, 2],
            clip_channels=1024)))

train_dataloader = dict(batch_size=8, num_workers=8, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=1, num_workers=1)