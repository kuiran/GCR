cudnn_benchmark = False
deterministic = True
seed = 1

num_stages = 2
num_proposals = 12

model = dict(
    type='UnifiedTrackersInput2',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4),
    rpn_head=dict(
        type='SinglePosAnchorRPNHead1',
        anchor_generator_cfg=dict(
            type='AnchorGenerator',
            strides=[4, 8, 16, 32],
            ratios=[0.5, 1.0, 2.0],
            scales=[8]),
        num_proposals=num_proposals,
        proposal_feature_channel=256,
        w_ratio=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        h_ratio=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    ),
    roi_head=dict(
        type='UnifiedTrackersInputRoIHead1',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        proposal_feature_channel=256,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        box_refinest1=dict(
            type='UTIDIIBoxHead',
            num_classes=1,
            num_ffn_fcs=2,
            num_heads=8,
            num_reg_fcs=3,
            feedforward_channels=2048,
            in_channels=256,
            reg_dims=4,
            dropout=0.0,
            ffn_act_cfg=dict(type='ReLU', inplace=True),
            dynamic_conv_cfg=dict(
                type='DynamicConv',
                in_channels=256,
                feat_channels=64,
                out_channels=256,
                input_feat_shape=7,
                act_cfg=dict(type='ReLU', inplace=True),
                norm_cfg=dict(type='LN')),
            loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            loss_iou=dict(type='GIoULoss', loss_weight=2.0),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                clip_border=False,
                target_means=[0., 0., 0., 0.],
                target_stds=[0.5, 0.5, 1., 1.])),
            # bbox_coder=dict(
            #     type='DistancePointBBoxCoder',
            #     clip_border=True
            # )
        pred_iou_head=dict(
                type='UTIDIIIOUHead',
                num_classes=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                feedforward_channels=2048,
                in_channels=256,
                reg_dims=4,
                dropout=0.0,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_pred_ious=dict(type='SmoothL1Loss', loss_weight=2.0)
        ),
        bbox_head=[
            dict(
                type='UTIDIIBoxHead',
                num_classes=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                reg_dims=4,
                dropout=0.0,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])
                # bbox_coder=dict(
                #     type='DistancePointBBoxCoder',
                #     clip_border=True
                # )
            ) for stage in range(num_stages)
        ],
        custom_cfg=dict(
            roi_connect=False,
            fix_proposal=True,
            num_proposals=num_proposals,
            use_text_feat=True,
            with_iterative_refine=True,
            with_prototype_selection=True,
            use_point_feat=False)
    ),
    # training and testing settings
    train_cfg=dict(
        rpn=None,
        rcnn=None,
        gen_point_factor=4
    ),
    test_cfg=dict(
        rpn=None,
        rcnn=None,
        vis_path=None,
        test_mode='point'
    )
)

data_root = 'data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='SingleSampling'),
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=False),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='GrayAug', prob=0.05),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='BrightnessAug', jitter_range=0.2),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'points']),
]
# test_pipeline = [
#     dict(type='TestSingleSampling'),
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True, with_label=False),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             # dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img', 'gt_bboxes', 'points', 'noisy_bbox']),
#         ])
# ]
img_norm_cfg1 = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_label=False),
    dict(
        type='MultiScaleFlipAugPointTrack',
        img_scale=(1333, 800),
        flip=False,
        transforms_frame0=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg1),
            dict(type='VideoCollect', keys=['img', 'gt_bboxes', 'points', 'noisy_bbox']),
            dict(type='ImageToTensor', keys=['img'])
        ],
        transforms_frame1=[
            dict(type='Normalize', **img_norm_cfg1),
            dict(type='VideoCollect', keys=['img', 'gt_bboxes', 'points', 'noisy_bbox']),
            dict(type='ImageToTensor', keys=['img'])
        ])
]
data_root = 'data/'
# dataset settings
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    persistent_workers=True,
    samples_per_epoch=64000,
    train=dict(
        type='RandomSampleConcatDataset',
        dataset_sampling_weights=[1],
        dataset_cfgs=[
            dict(
                type='LaSOTPointDataset',
                point_prefix_test=data_root + 'lasot/lasot_test_ctr_1',
                point_prefix_train=data_root + 'lasot/lasot_train_points',
                only_load_first_frame=False,
                noisy_bbox_prefix=data_root + 'lasot/lasot_test_noise',
                with_track_bbox=False,
                ann_file=data_root + 'lasot/annotations/lasot_train_infos.txt',
                img_prefix=data_root + 'lasot/LaSOTBenchmark',
                pipeline=train_pipeline,
                split='train',
                test_mode=False
            )
        ]
    ),
    val=dict(
        type='LaSOTPointDataset',
        point_prefix_test=data_root + 'lasot/lasot_test_ctr_1',
        point_prefix_train=data_root + 'lasot/lasot_train_points',
        only_load_first_frame=False,
        noisy_bbox_prefix=data_root + 'lasot/lasot_test_noise',
        with_track_bbox=False,
        ann_file=data_root + 'lasot/annotations/lasot_test_infos.txt',
        img_prefix=data_root + 'lasot/LaSOTBenchmark',
        pipeline=test_pipeline,
        split='test',
        test_mode=True),
    test=dict(
        type='LaSOTPointDataset',
        point_prefix_test=data_root + 'lasot/lasot_test_ctr_1',
        point_prefix_train=data_root + 'lasot/lasot_train_points',
        only_load_first_frame=True,
        # noisy_bbox_prefix=data_root + 'lasot/lasot_test_noise',
        noisy_bbox_prefix=data_root + 'lasot/lasot_test_noise_0.1to0.2',
        with_track_bbox=False,
        ann_file=data_root + 'lasot/annotations/lasot_test_infos.txt',
        img_prefix=data_root + 'lasot/LaSOTBenchmark',
        pipeline=test_pipeline,
        split='test',
        test_mode=True
     )
)
#
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     persistent_workers=True,
#     samples_per_epoch=60000,
#     train=dict(
#         type='RandomSampleConcatDataset',
#         dataset_sampling_weights=[1],
#         dataset_cfgs=[
#             dict(
#                 type='LaSOTPointDataset',
#                 point_prefix_test='/mnt/dataset/data/lasot/lasot_test_ctr_1',
#                 point_prefix_train='/mnt/dataset/data/lasot/lasot_train_points',
#                 only_load_first_frame=False,
#                 noisy_bbox_prefix='/home/ubuntu/mnt/dataset/data/lasot/lasot_test_noise',
#                 with_track_bbox=False,
#                 ann_file=data_root + 'lasot/annotations/lasot_train_infos.txt',
#                 img_prefix=data_root + 'lasot/LaSOTBenchmark',
#                 pipeline=train_pipeline,
#                 split='train',
#                 test_mode=False
#             ),
#             # dict(
#             #     type='SOTCocoDataset',
#             #     ann_file=data_root +
#             #     'coco/coco/annotations/instances_train2017_point.json',
#             #     img_prefix=data_root + 'coco/coco/train2017',
#             #     pipeline=train_pipeline,
#             #     split='train',
#             #     test_mode=False)
#         ]
#     ),
#     val=dict(
#         type='LaSOTPointDataset',
#         point_prefix_test=data_root + 'lasot/lasot_test_ctr_1',
#         point_prefix_train=data_root + 'lasot/lasot_train_points',
#         only_load_first_frame=False,
#         noisy_bbox_prefix=data_root + 'lasot/lasot_test_noise',
#         with_track_bbox=False,
#         ann_file=data_root + 'lasot/annotations/lasot_test_infos.txt',
#         img_prefix=data_root + 'lasot/LaSOTBenchmark',
#         pipeline=test_pipeline,
#         split='test',
#         test_mode=True),
#     test=dict(
#         type='LaSOTPointDataset',
#         point_prefix_test=data_root + 'lasot/lasot_test_ctr_1',
#         point_prefix_train=data_root + 'lasot/lasot_train_points',
#         only_load_first_frame=False,
#         # noisy_bbox_prefix=data_root + 'lasot/lasot_test_noise',
#         noisy_bbox_prefix=data_root + 'lasot/lasot_test_noise_0.3to0.4',
#         with_track_bbox=False,
#         ann_file=data_root + 'lasot/annotations/lasot_test_infos.txt',
#         img_prefix=data_root + 'lasot/LaSOTBenchmark',
#         pipeline=test_pipeline,
#         split='test',
#         test_mode=True
#     # test=dict(
#     #     type='GOT10kPointDataset',
#     #     ann_file=data_root + 'got10k/annotations/got10k_test_infos.txt',
#     #     img_prefix=data_root + 'got10k',
#     #     pipeline=test_pipeline,
#     #     split='test',
#     #     test_mode=True)
#      )
# )
# optimizer
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
# log
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
log_level = 'INFO'
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
# checkpoint saving
checkpoint_config = dict(interval=3)

evaluation = dict(
    metric=['track'],
    interval=100,
    start=501,
    rule='greater',
    save_best='success')

load_from = None
resume_from = None
total_epochs = 12
work_dir = './work_dirs/xxx'
