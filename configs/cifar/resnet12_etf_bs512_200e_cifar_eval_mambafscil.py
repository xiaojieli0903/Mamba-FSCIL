_base_ = [
    '../_base_/models/resnet_etf.py', '../_base_/datasets/cifar_fscil.py',
    '../_base_/schedules/cifar_200e.py', '../_base_/default_runtime.py'
]

model = dict(neck=dict(type='MambaNeck',
                       version='ssm',
                       in_channels=640,
                       out_channels=640,
                       feat_size=2,
                       num_layers=2,
                       use_residual_proj=False,
                       use_new_branch=True,
                       detach_residual=True,
                       num_layers_new=2,
                       loss_weight_supp=100,
                       loss_weight_supp_novel=0.00001,
                       loss_weight_sep_new=0.001,
                       param_avg_dim='0-1-3'),
             head=dict(type='ETFHead', in_channels=640, with_len=True),
             mixup=0.5,
             mixup_prob=0.75)

copy_list = (1, 1, 1, 1, 1, 1, 1, 1, None, None)
step_list = (200, 200, 200, 200, 200, 200, 200, 200, None, None)

finetune_lr = 0.25

# optimizer
optimizer = dict(type='SGD',
                 lr=finetune_lr,
                 momentum=0.9,
                 weight_decay=0.0005,
                 paramwise_cfg=dict(
                     custom_keys={
                         'neck.mlp_proj.': dict(lr_mult=0),
                         'neck.block.': dict(lr_mult=0),
                         'neck.residual_proj.': dict(lr_mult=0),
                         'neck.pos_embed': dict(lr_mult=0),
                         'neck.pos_embed_new': dict(lr_mult=1)
                     }))

find_unused_parameters = True
