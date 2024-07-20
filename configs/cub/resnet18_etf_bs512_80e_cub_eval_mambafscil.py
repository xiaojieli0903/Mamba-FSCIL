_base_ = [
    '../_base_/models/resnet_etf.py', '../_base_/datasets/cub_fscil.py',
    '../_base_/schedules/cub_80e.py', '../_base_/default_runtime.py'
]

# CUB requires different inc settings
inc_start = 100
inc_end = 200
inc_step = 10

# model settings
model = dict(backbone=dict(_delete_=True,
                           type='ResNet',
                           depth=18,
                           frozen_stages=1,
                           init_cfg=dict(type='Pretrained',
                                         checkpoint='torchvision://resnet18'),
                           norm_cfg=dict(type='BN', requires_grad=True)),
             neck=dict(type='MambaNeck',
                       version='ss2d',
                       in_channels=512,
                       out_channels=1024,
                       feat_size=7,
                       num_layers=2,
                       use_residual_proj=True,
                       use_new_branch=True,
                       detach_residual=False,
                       num_layers_new=3,
                       loss_weight_supp=100,
                       loss_weight_supp_novel=10,
                       loss_weight_sep=0.001,
                       loss_weight_sep_new=0.001,
                       param_avg_dim='0-1-3'),
             head=dict(type='ETFHead',
                       in_channels=1024,
                       num_classes=200,
                       eval_classes=100,
                       with_len=False),
             mixup=0.5,
             mixup_prob=0.5)

base_copy_list = (1, 1, 2, 2, 3, 3, 1, 1, 1, 1)
copy_list = (10, 10, 10, 10, 10, 10, 10, 10, 10, 10)
step_list = (200, 210, 220, 230, 240, 250, 260, 270, 280, 290)
finetune_lr = 0.05

# optimizer
optimizer = dict(type='SGD',
                 lr=finetune_lr,
                 momentum=0.9,
                 weight_decay=0.0005,
                 paramwise_cfg=dict(
                     custom_keys={
                         'neck.mlp_proj.': dict(lr_mult=0.2),
                         'neck.block.': dict(lr_mult=0.2),
                         'neck.residual_proj': dict(lr_mult=0.2),
                         'neck.pos_embed': dict(lr_mult=0.2),
                         'neck.pos_embed_new': dict(lr_mult=1)
                     }))

optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='CosineAnnealingCooldown',
    min_lr=None,
    min_lr_ratio=0.1,
    cool_down_ratio=0.1,
    cool_down_time=10,
    by_epoch=False,
    # warmup
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.1,
    warmup_by_epoch=False)
