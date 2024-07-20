from collections import OrderedDict

import torch.nn as nn
from mmcv.cnn import build_norm_layer

from mmcls.models.builder import NECKS


@NECKS.register_module()
class MLPFFNNeck(nn.Module):

    def __init__(self,
                 in_channels=512,
                 out_channels=512,
                 num_layers=3,
                 drop_rate=0,
                 use_final_residual=True,
                 forward_main=True,
                 loss_weight_main=0,
                 loss_weight_main_novel=0,
                 repeat_branch=1):
        super().__init__()
        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(drop_rate)
        self.use_final_residual = use_final_residual
        self.forward_main = forward_main
        self.loss_weight_main = loss_weight_main
        self.loss_weight_main_novel = loss_weight_main_novel
        self.repeat_branch = repeat_branch

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        if self.num_layers == 3:
            self.ln1 = nn.Sequential(
                OrderedDict([('linear', nn.Linear(in_channels,
                                                  in_channels * 2)),
                             ('ln',
                              build_norm_layer(dict(type='LN'),
                                               in_channels * 2)[1]),
                             ('relu', nn.LeakyReLU(0.1))]))
            self.ln2 = nn.Sequential(
                OrderedDict([
                    ('linear', nn.Linear(in_channels * 2, in_channels * 2)),
                    ('ln', build_norm_layer(dict(type='LN'),
                                            in_channels * 2)[1]),
                    ('relu', nn.LeakyReLU(0.1))
                ]))
            self.ln3 = nn.Sequential(
                OrderedDict([
                    ('linear',
                     nn.Linear(in_channels * 2, out_channels, bias=False)),
                ]))
        elif self.num_layers == 2:
            self.ln1 = nn.Sequential(
                OrderedDict([('linear', nn.Linear(in_channels,
                                                  in_channels * 2)),
                             ('ln',
                              build_norm_layer(dict(type='LN'),
                                               in_channels * 2)[1]),
                             ('relu', nn.LeakyReLU(0.1))]))
            self.ln3 = nn.Sequential(
                OrderedDict([
                    ('linear',
                     nn.Linear(in_channels * 2, out_channels, bias=False)),
                ]))
        elif self.num_layers == 1:
            self.ln1 = nn.Sequential(
                OrderedDict([('linear', nn.Linear(in_channels, out_channels)),
                             ('ln',
                              build_norm_layer(dict(type='LN'),
                                               out_channels)[1]),
                             ('relu', nn.LeakyReLU(0.1))]))

        if in_channels == out_channels:
            self.ffn = nn.Sequential(
                OrderedDict([
                    ('proj', nn.Linear(in_channels, out_channels, bias=False)),
                ]))
        else:
            self.ffn = nn.Sequential(
                OrderedDict([
                    ('proj', nn.Linear(in_channels, out_channels, bias=False)),
                ]))

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            inputs = inputs[-1]
        x = self.avg(inputs)
        x = x.view(inputs.size(0), -1)
        identity = x
        outputs = {}
        if self.num_layers == 3:
            x = self.ln1(x)
            if self.drop_rate > 0:
                x = self.drop_layer(x)
            x = self.ln2(x)
            x = self.ln3(x)
            identity = self.ffn(identity)
            outputs['main'] = x
            outputs['residual'] = identity
            if self.use_final_residual:
                x = x + identity
        elif self.num_layers == 2:
            x = self.ln1(x)
            if self.drop_rate > 0:
                x = self.drop_layer(x)
            x = self.ln3(x)
            identity = self.ffn(identity)
            outputs['main'] = x
            outputs['residual'] = identity
            if self.use_final_residual:
                x = x + identity
        elif self.num_layers == 1:
            x = self.ln1(x)
            if self.drop_rate > 0:
                x = self.drop_layer(x)
            identity = self.ffn(identity)
            outputs['main'] = x
            outputs['residual'] = identity
            if self.use_final_residual:
                x = x + identity
        elif self.num_layers == 0:
            x = self.ffn(x)
            outputs['main'] = x
            outputs['residual'] = None
        outputs['out'] = x
        return outputs
