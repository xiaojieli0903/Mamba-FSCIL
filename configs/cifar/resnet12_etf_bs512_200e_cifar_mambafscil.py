_base_ = [
    '../_base_/models/resnet_etf.py', '../_base_/datasets/cifar_fscil.py',
    '../_base_/schedules/cifar_200e.py', '../_base_/default_runtime.py'
]

model = dict(
    neck=dict(type='MambaNeck',
              version='ssm',
              in_channels=640,
              out_channels=640,
              feat_size=2,
              num_layers=2,
              use_residual_proj=False),
    head=dict(type='ETFHead', in_channels=640, with_len=False),
    train_cfg=dict(augments=[
        dict(type='BatchMixupTwoLabel', alpha=0.8, num_classes=-1, prob=0.4),
        dict(type='BatchCutMixTwoLabel', alpha=1.0, num_classes=-1, prob=0.4),
        dict(type='IdentityTwoLabel', num_classes=-1, prob=0.2),
    ]),
    mixup=0,
    mixup_prob=0)
