from collections import OrderedDict

import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from rope import *
from timm.models.layers import trunc_normal_

from mmcls.models.builder import NECKS
from mmcls.utils import get_root_logger

from .mamba_ssm.modules.mamba_simple import Mamba
from .ss2d import SS2D


@NECKS.register_module()
class MambaNeck(BaseModule):
    """Dual selective SSM branch in Mamba-FSCIL framework.

        This module integrates our dual selective SSM branch for dynamic adaptation in few-shot
        class-incremental learning tasks.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            mid_channels (int, optional): Number of intermediate channels in MLP projections, defaults to twice the in_channels if not specified.
            version (str): Specifies the version of the state space model; 'ssm' or 'ss2d'.
            use_residual_proj (bool): If True, adds a residual projection.
            d_state (int): Dimension of the hidden state in the SSM.
            d_rank (int, optional): Dimension rank in the SSM, if not provided, defaults to d_state.
            ssm_expand_ratio (float): Expansion ratio for the SSM block.
            num_layers (int): Number of layers in the MLP projections.
            num_layers_new (int, optional): Number of layers in the new branch MLP projections, defaults to num_layers if not specified.
            feat_size (int): Size of the input feature map.
            use_new_branch (bool): If True, uses an additional branch for incremental learning.
            loss_weight_supp (float): Loss weight for suppression term for base classes.
            loss_weight_supp_novel (float): Loss weight for suppression term for novel classes.
            loss_weight_sep (float): Loss weight for separation term during the base session.
            loss_weight_sep_new (float): Loss weight for separation term during the incremental session.
            param_avg_dim (str): Dimensions to average for computing averaged input-dependment parameter features; '0-1' or '0-3' or '0-1-3'.
            detach_residual (bool): If True, detaches the residual connections during the output computation.
    """

    def __init__(self,
                 in_channels=512,
                 out_channels=512,
                 mid_channels=None,
                 version='ssm',
                 use_residual_proj=False,
                 d_state=256,
                 d_rank=None,
                 ssm_expand_ratio=1,
                 num_layers=2,
                 num_layers_new=None,
                 feat_size=2,
                 use_new_branch=False,
                 loss_weight_supp=0.0,
                 loss_weight_supp_novel=0.0,
                 loss_weight_sep=0.0,
                 loss_weight_sep_new=0.0,
                 param_avg_dim='0-1-3',
                 detach_residual=False):
        super(MambaNeck, self).__init__(init_cfg=None)
        self.version = version
        assert self.version in ['ssm', 'ss2d'], f'Invalid branch version.'
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.use_residual_proj = use_residual_proj
        self.mid_channels = in_channels * 2 if mid_channels is None else mid_channels
        self.feat_size = feat_size
        self.d_state = d_state
        self.d_rank = d_rank if d_rank is not None else d_state
        self.use_new_branch = use_new_branch
        self.num_layers = num_layers
        self.num_layers_new = self.num_layers if num_layers_new is None else num_layers_new
        self.detach_residual = detach_residual
        self.loss_weight_supp = loss_weight_supp
        self.loss_weight_supp_novel = loss_weight_supp_novel
        self.loss_weight_sep = loss_weight_sep
        self.loss_weight_sep_new = loss_weight_sep_new
        self.param_avg_dim = [int(item) for item in param_avg_dim.split('-')]
        self.logger = get_root_logger()
        directions = ('h', 'h_flip', 'v', 'v_flip')

        # Positional embeddings for features
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.feat_size * self.feat_size, out_channels))
        trunc_normal_(self.pos_embed, std=.02)

        if self.use_new_branch:
            self.pos_embed_new = nn.Parameter(
                torch.zeros(1, self.feat_size * self.feat_size, out_channels))
            trunc_normal_(self.pos_embed_new, std=.02)

        if self.num_layers == 3:
            self.mlp_proj = self.build_mlp(in_channels,
                                           out_channels,
                                           self.mid_channels,
                                           num_layers=3)
        elif self.num_layers == 2:
            self.mlp_proj = self.build_mlp(in_channels,
                                           out_channels,
                                           self.mid_channels,
                                           num_layers=2)

        if self.version == 'ssm':
            self.block = Mamba(out_channels,
                               expand=ssm_expand_ratio,
                               use_out_proj=False,
                               d_state=d_state,
                               dt_rank=self.d_rank)
        else:
            self.block = SS2D(out_channels,
                              ssm_ratio=ssm_expand_ratio,
                              d_state=d_state,
                              dt_rank=self.d_rank,
                              directions=directions,
                              use_out_proj=False,
                              use_out_norm=True)

        if self.use_new_branch:
            if self.num_layers_new == 3:
                self.mlp_proj_new = self.build_mlp(in_channels,
                                                   out_channels,
                                                   self.mid_channels,
                                                   num_layers=3)
            elif self.num_layers_new == 2:
                self.mlp_proj_new = self.build_mlp(in_channels,
                                                   out_channels,
                                                   self.mid_channels,
                                                   num_layers=2)

            if self.version == 'ssm':
                self.block_new = Mamba(out_channels,
                                       expand=ssm_expand_ratio,
                                       use_out_proj=False,
                                       d_state=d_state,
                                       dt_rank=self.d_rank)
            else:
                self.block_new = SS2D(out_channels,
                                      ssm_ratio=ssm_expand_ratio,
                                      d_state=d_state,
                                      dt_rank=self.d_rank,
                                      directions=directions,
                                      use_out_proj=False,
                                      use_out_norm=True)

        if self.use_residual_proj:
            self.residual_proj = nn.Sequential(
                OrderedDict([
                    ('proj', nn.Linear(in_channels, out_channels, bias=False)),
                ]))

        self.init_weights()

    def build_mlp(self, in_channels, out_channels, mid_channels, num_layers):
        """Builds the MLP projection part of the neck.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            mid_channels (int): Number of mid-level channels.
            num_layers (int): Number of linear layers in the MLP.

        Returns:
            nn.Sequential: The MLP layers as a sequential module.
        """
        layers = []
        layers.append(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=1,
                      padding=0,
                      bias=True))
        layers.append(
            build_norm_layer(
                dict(type='LN'),
                [mid_channels, self.feat_size, self.feat_size])[1])
        layers.append(nn.LeakyReLU(0.1))

        if num_layers == 3:
            layers.append(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=1,
                          bias=True))
            layers.append(
                build_norm_layer(
                    dict(type='LN'),
                    [mid_channels, self.feat_size, self.feat_size])[1])
            layers.append(nn.LeakyReLU(0.1))

        layers.append(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False))
        return nn.Sequential(*layers)

    def init_weights(self):
        """Zero initialization for the newly attached residual branche."""
        if self.use_new_branch:
            with torch.no_grad():
                dim_proj = int(self.block_new.in_proj.weight.shape[0] / 2)
                self.block_new.in_proj.weight.data[-dim_proj:, :].zero_()
            self.logger.info(
                f'--MambaNeck zero_init_residual z: '
                f'(self.block_new.in_proj.weight{self.block_new.in_proj.weight.shape}), '
                f'{torch.norm(self.block_new.in_proj.weight.data[-dim_proj:, :])}'
            )

    def forward(self, x):
        """Forward pass for MambaNeck, integrating both the main and an optional new branch for processing.

            Args:
                x (Tensor): Input tensor, potentially as part of a tuple from previous layers.

            Returns:
                dict: A dictionary of outputs including processed features from main and new branches,
                      along with the combined final output.
            """
        # Extract the last element if input is a tuple (from previous layers).
        if isinstance(x, tuple):
            x = x[-1]
        B, C, H, W = x.shape
        identity = x
        outputs = {}

        C, dts, Bs, Cs, C_new, dts_new, Bs_new, Cs_new = None, None, None, None, None, None, None, None

        if self.detach_residual:
            self.block.eval()
            self.mlp_proj.eval()

        # Prepare the identity projection for the residual connection
        if self.use_residual_proj:
            identity_proj = self.residual_proj(self.avg(identity).view(B, -1))
        else:
            identity_proj = self.avg(identity).view(B, -1)
        x = self.mlp_proj(identity).permute(0, 2, 3, 1).view(B, H * W, -1)

        # Process the input tensor through MLP projection and add positional embeddings
        x = x.view(B, H * W, -1) + self.pos_embed

        # First selective SSM branch processing
        if self.version == 'ssm':
            # SSM block processing
            x_h, C_h = self.block(x, return_param=True)
            if isinstance(C_h, list):
                C_h, dts, Bs, Cs = C_h
                outputs.update({
                    'dts':
                    dts.view(dts.shape[0], 1, dts.shape[1], dts.shape[2]),
                    'Bs':
                    Bs.view(Bs.shape[0], 1, Bs.shape[1], Bs.shape[2]),
                    'Cs':
                    Cs.view(Cs.shape[0], 1, Cs.shape[1], Cs.shape[2])
                })
            # Handle horizontal and vertical symmetry by processing flipped versions.
            x_hf, C_hf = self.block(x.flip([1]), return_param=False)
            xs_v = rearrange(x, 'b (h w) d -> b (w h) d', h=H,
                             w=W).view(B, H * W, -1)
            x_v, C_v = self.block(xs_v, return_param=False)
            x_vf, C_vf = self.block(xs_v.flip([1]), return_param=False)

            x = x_h + x_hf.flip([1]) + rearrange(
                x_v, 'b (h w) d -> b (w h) d', h=H, w=W) + rearrange(
                    x_vf.flip([1]), 'b (h w) d -> b (w h) d', h=H, w=W)
            C = C_h + C_hf.flip([1]) + rearrange(
                C_v, 'b d (h w) -> b d (w h)', h=H, w=W) + rearrange(
                    C_vf.flip([1]), 'b d (h w) -> b d (w h)', h=H, w=W)
            x = self.avg(x.permute(0, 2, 1).reshape(B, -1, H, W)).view(B, -1)
        else:
            # SS2D processing
            x = x.view(B, H, W, -1)
            x, C = self.block(x, return_param=True)

            if isinstance(C, list):
                C, dts, Bs, Cs = C
                outputs.update({'dts': dts, 'Bs': Bs, 'Cs': Cs})
            x = self.avg(x.permute(0, 3, 1, 2)).view(B, -1)

        # New branch processing for incremental learning sessions, if enabled.
        if self.use_new_branch:
            x_new = self.mlp_proj_new(identity.detach()).permute(
                0, 2, 3, 1).view(B, H * W, -1)
            x_new += self.pos_embed_new
            if self.version == 'ssm':
                x_h_new, C_h_new = self.block_new(x_new, return_param=True)
                if isinstance(C_h_new, list):
                    C_h_new, dts_new, Bs_new, Cs_new = C_h_new
                    outputs.update({
                        'dts_new':
                        dts_new.view(dts_new.shape[0], 1, dts_new.shape[1],
                                     dts_new.shape[2]),
                        'Bs_new':
                        Bs_new.view(Bs_new.shape[0], 1, Bs_new.shape[1],
                                    Bs_new.shape[2]),
                        'Cs_new':
                        Cs_new.view(Cs_new.shape[0], 1, Cs_new.shape[1],
                                    Cs_new.shape[2])
                    })

                x_hf_new, C_hf_new = self.block_new(x_new.flip([1]),
                                                    return_param=False)
                xs_v_new = rearrange(x_new, 'b (h w) d -> b (w h) d', h=H,
                                     w=W).view(B, H * W, -1)
                x_v_new, C_v_new = self.block_new(xs_v_new, return_param=False)
                x_vf_new, C_vf_new = self.block_new(xs_v_new.flip([1]),
                                                    return_param=False)

                # Combine outputs from new branch.
                x_new = x_h_new + x_hf_new.flip([1]) + rearrange(
                    x_v_new, 'b (h w) d -> b (w h) d', h=H, w=W) + rearrange(
                        x_vf_new.flip([1]), 'b (h w) d -> b (w h) d', h=H, w=W)
                C_new = C_h_new + C_hf_new.flip([1]) + rearrange(
                    C_v_new, 'b d (h w) -> b d (w h)', h=H, w=W) + rearrange(
                        C_vf_new.flip([1]), 'b d (h w) -> b d (w h)', h=H, w=W)
                x_new = self.avg(x_new.permute(0, 2,
                                               1).reshape(B, -1, H,
                                                          W)).view(B, -1)
            else:
                x_new = x_new.view(B, H, W, -1)
                x_new, C_new = self.block_new(x_new, return_param=True)
                if isinstance(C_new, list):
                    C_new, dts_new, Bs_new, Cs_new = C_new
                    outputs.update({
                        'dts_new': dts_new,
                        'Bs_new': Bs_new,
                        'Cs_new': Cs_new
                    })
                x_new = self.avg(x_new.permute(0, 3, 1, 2)).view(B, -1)
        """Combines outputs from the main and new branches with the identity projection."""
        if not self.use_new_branch:
            outputs['main'] = C if C is not None else x
            outputs['residual'] = identity_proj
            x = x + identity_proj
        else:
            outputs['main'] = C_new if C_new is not None else x_new
            outputs['residual'] = x + identity_proj
            if self.detach_residual:
                x = x.detach() + identity_proj.detach() + x_new
            else:
                x = x + identity_proj + x_new

        outputs['out'] = x
        return outputs
