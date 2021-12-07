_base_ = [
    "./_base_/detector/cascade_rcnn_fpn_bbox.py",
    "./_base_/env/coco_instance.py",
    "./_base_/env/schedule_1x.py",
    "./_base_/env/default_runtime.py",
]

# modify coco_instance.py
classes = (
    "01_ulcer",
    "02_mass",
    "04_lymph",
    "05_bleeding",
)
num_classes = 4
data_root = "/home/haejin/work/dataset/coco"
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Rotate", prob=1, level=10, replace=(0, 90, 180, 270)),
    dict(
        type="AutoAugment",
        policies=[
            [
                dict(
                    type="Resize",
                    img_scale=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    multiscale_mode="value",
                    keep_ratio=True,
                )
            ],
            [
                dict(
                    type="Resize",
                    img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                    multiscale_mode="value",
                    keep_ratio=True,
                ),
                dict(
                    type="RandomCrop",
                    crop_type="absolute_range",
                    crop_size=(384, 600),
                    allow_negative_crop=True,
                ),
                dict(
                    type="Resize",
                    img_scale=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    multiscale_mode="value",
                    override=True,
                    keep_ratio=True,
                ),
            ],
        ],
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

evaluation = dict(metric=["bbox"], save_best="bbox_mAP", interval=1)

# modify schedule_1x.py
optimizer = dict(
    _delete_=True, type="AdamW", lr=4e-5, betas=(0.9, 0.999), weight_decay=0.05
)  # paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
#                                 'relative_position_bias_table': dict(decay_mult=0.),
#                                 'norm': dict(decay_mult=0.)}))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    _delete_=True,
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=4e-6,
)
runner = dict(type="EpochBasedRunner", max_epochs=40)

# modify default_runtime.py
log_config = dict(hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")])
workflow = [("train", 1), ("val", 1)]


# model
pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"  # noqa

model = dict(
    backbone=dict(
        type="SwinTransformer",
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
    # detector
    # roi_head=dict(
    #     bbox_head=dict(
    #         num_classes=num_classes
    #         ),
    #     )
)
