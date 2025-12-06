
max_epochs=300

model = dict(type='TrainYOLO11',
    backbone_config=dict(
        model_type = "n",
        num_classes = 80),
    loss_config=dict(
        strides = [8, 16, 32],
        num_classes = 80,
        dfl_bins = 16,
        loss_cls_weight = 1.,
        loss_iou_weight = 7.5,
        loss_dfl_weight = 1.5,
        tal_topk = 10,
        use_focal = True,          # focal for BCE path
        focal_alpha = 0.25,
        focal_gamma = 1.0,
        debug = False,
        # hard-neg 采样和正负样本权重
        neg_pos_ratio = 1,
        neg_thresh = 0.05,
        pos_weight = 3.0,
        neg_weight = 0.5,
        # new control params
        pos_loss_type = 'bce',   # 'bce' (default) or 'ce' (softmax CE focal)
        use_neg_loss = True,
        neg_selection = 'per_image',
        label_smoothing = 0.0,
        o2m_weight = .8))

work_dir = '/lpai/volumes/vc-profile-bd-ga/others/wubo/Projects/Code/011-computer-version/myyolo/work_dirs/yolo11'

train_dataloader = dict(
    dataset=dict(type='YOLOCOCO',
        root = "/lpai/volumes/vc-profile-bd-ga/others/yanxiaobao/datasets/COCO2017/train2017",
        ann_file = "/lpai/volumes/vc-profile-bd-ga/others/yanxiaobao/datasets/COCO2017/annotations/instances_train2017.json"),
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='coco_collate'),
    batch_size=64,
    drop_last=True,
    pin_memory=True,
    persistent_workers=True,
    num_workers=4)

train_cfg = dict(
    by_epoch=True,
    max_epochs=max_epochs,
    val_begin=1,
    val_interval=1)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=4e-3,
        weight_decay=1e-2))

param_scheduler = [
    dict(  # 预热
        type='LinearLR',
        start_factor=.001,
        by_epoch=True,
        begin=0,
        end=6,
        verbose=True),
    dict(  # 阶梯下降
        type='MultiStepLR',
        by_epoch=True,
        milestones=[60, 120, 180],
        gamma=0.1)]


# val_dataloader = dict(
#     dataset=dict(type='YOLOCOCO',
#         root = "/lpai/volumes/vc-profile-bd-ga/others/yanxiaobao/datasets/COCO2017/val2017",
#         ann_file = "/lpai/volumes/vc-profile-bd-ga/others/yanxiaobao/datasets/COCO2017/annotations/instances_val2017.json"),
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     collate_fn=dict(type='default_collate'),
#     batch_size=128,
#     drop_last=False,
#     pin_memory=False,
#     persistent_workers=False,
#     num_workers=8)
# val_evaluator = dict(type='GenderAgeMetric')

# val_cfg = dict(type='ValLoop')

default_hooks = dict(
    logger=dict(type='LoggerHook',
        interval=10,
        log_metric_by_epoch=True),
    checkpoint=dict(type='CheckpointHook', interval=1))

launcher = 'pytorch'

env_cfg = dict(
    cudnn_benchmark=False,
    backend='nccl',
    mp_cfg=dict(mp_start_method='spawn'),
    dist_cfg=dict(backend='nccl'))

log_level = 'INFO'

load_from = None

resume = False