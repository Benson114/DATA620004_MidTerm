_base_ = [
    '../_base_/models/rpn_r50_fpn.py',
    # '../_base_/datasets/coco_detection.py',
    '../_base_/datasets/voc0712_lzf.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# val_evaluator = dict(metric='proposal_fast')
val_evaluator = dict(metric='mAP')
test_evaluator = val_evaluator
model = dict(
    # 从 configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py 复制
    neck=dict(
        start_level=1,
        add_extra_convs='on_output',  # 使用 P5
        relu_before_extra_convs=True),
    rpn_head=dict(
        _delete_=True,  # 忽略未使用的旧设置
        type='FCOSHead',
        num_classes=1,  # 对于 rpn, num_classes = 1，如果 num_classes >为1，它将在 rpn 中自动设置为1
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))