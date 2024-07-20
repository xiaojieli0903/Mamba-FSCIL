_base_ = [
    '../_base_/models/resnet_etf.py',
    '../_base_/datasets/mini_imagenet_fscil.py',
    '../_base_/schedules/mini_imagenet_500e.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(neck=dict(type='MambaNeck',
                       version='ssm',
                       in_channels=640,
                       out_channels=512,
                       feat_size=5,
                       num_layers=2,
                       use_residual_proj=True,
                       use_new_branch=True,
                       detach_residual=True,
                       num_layers_new=2,
                       loss_weight_supp=100,
                       loss_weight_sep_new=0.5,
                       param_avg_dim='0-1-3'),
             head=dict(type='ETFHead', in_channels=512, with_len=True),
             mixup=0.5,
             mixup_prob=0.3)

base_copy_list = (1, 2, 3, 4, 5, 6, 7, 8, None, None)
step_list = (100, 110, 120, 130, 140, 150, 160, 170, None, None)
copy_list = (10, 10, 10, 10, 10, 10, 10, 10, None, None)

finetune_lr = 0.01

# optimizer
optimizer = dict(type='SGD',
                 lr=finetune_lr,
                 momentum=0.9,
                 weight_decay=0.0005,
                 paramwise_cfg=dict(
                     custom_keys={
                         'neck.block.': dict(lr_mult=0.0),
                         'neck.residual_proj': dict(lr_mult=0.0),
                         'neck.pos_embed': dict(lr_mult=0.0),
                         'neck.mlp_proj.': dict(lr_mult=0.0),
                         'neck.pos_embed_new': dict(lr_mult=1)
                     }))
find_unused_parameters = True
