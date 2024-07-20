import math

import selective_scan_cuda
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def multi_selective_scan(x: torch.Tensor = None,
                         x_proj_weight: torch.Tensor = None,
                         x_proj_bias: torch.Tensor = None,
                         dt_projs_weight: torch.Tensor = None,
                         dt_projs_bias: torch.Tensor = None,
                         A_logs: torch.Tensor = None,
                         Ds: torch.Tensor = None,
                         out_norm: torch.nn.Module = None,
                         nrows=-1,
                         delta_softplus=True,
                         to_dtype=True,
                         multi_scan=None,
                         return_param=False):
    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows < 1:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    xs = multi_scan.multi_scan(x)

    L = xs.shape[-1]
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs,
                         x_proj_weight)  # l fixed

    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    if return_param:
        output = [dts, Bs, Cs]
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    xs = xs.view(B, -1, L).to(torch.float)
    dts = dts.contiguous().view(B, -1, L).to(torch.float)
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous().to(torch.float)
    Cs = Cs.contiguous().to(torch.float)
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    def selective_scan(u,
                       delta,
                       A,
                       B,
                       C,
                       D=None,
                       delta_bias=None,
                       delta_softplus=True,
                       nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias,
                                   delta_softplus, nrows)

    ys: torch.Tensor = selective_scan(
        xs,
        dts,
        As,
        Bs,
        Cs,
        Ds,
        delta_bias,
        delta_softplus,
        nrows,
    ).view(B, K, -1, L)

    y = multi_scan.merge(ys)

    y = out_norm(y).view(B, H, W, -1)

    if not return_param:
        return (y.to(x.dtype) if to_dtype else y)
    else:
        return (y.to(x.dtype) if to_dtype else y), output


class MultiScan(nn.Module):
    ALL_CHOICES = ('h', 'h_flip', 'v', 'v_flip')

    def __init__(self, dim, choices=None, token_size=(14, 14)):
        super().__init__()
        self.token_size = token_size
        self.choices = choices
        self.search = False

    def forward(self, xs):
        """
        Input @xs: [[B, L, D], ...]
        """
        x = torch.stack(xs).sum(0)
        return x

    def multi_scan(self, x):
        """
        Input @x: shape [B, L, D]
        """
        xs = []
        for direction in self.choices:
            xs.append(self.scan(x, direction))
        return xs

    def multi_reverse(self, xs):
        new_xs = []
        for x, direction in zip(xs, self.choices):
            new_xs.append(self.reverse(x, direction))
        return new_xs

    def scan(self, x, direction='h'):
        """
        Input @x: shape [B, L, D] or [B, C, H, W]
        Return torch.Tensor: shape [B, D, L]
        """
        H, W = self.token_size
        if len(x.shape) == 3:
            if direction == 'h':
                return x.transpose(-2, -1)
            elif direction == 'h_flip':
                return x.transpose(-2, -1).flip([-1])
            elif direction == 'v':
                return rearrange(x, 'b (h w) d -> b d (w h)', h=H, w=W)
            elif direction == 'v_flip':
                return rearrange(x, 'b (h w) d -> b d (w h)', h=H,
                                 w=W).flip([-1])
            else:
                raise RuntimeError(f'Direction {direction} not found.')
        elif len(x.shape) == 4:
            if direction == 'h':
                return x.flatten(2)
            elif direction == 'h_flip':
                return x.flatten(2).flip([-1])
            elif direction == 'v':
                return rearrange(x, 'b d h w -> b d (w h)', h=H, w=W)
            elif direction == 'v_flip':
                return rearrange(x, 'b d h w -> b d (w h)', h=H,
                                 w=W).flip([-1])
            else:
                raise RuntimeError(f'Direction {direction} not found.')

    def reverse(self, x, direction='h'):
        """
        Input @x: shape [B, D, L]
        Return torch.Tensor: shape [B, D, L]
        """
        H, W = self.token_size
        if direction == 'h':
            return x
        elif direction == 'h_flip':
            return x.flip([-1])
        elif direction == 'v':
            return rearrange(x, 'b d (h w) -> b d (w h)', h=H, w=W)
        elif direction == 'v_flip':
            return rearrange(x.flip([-1]), 'b d (h w) -> b d (w h)', h=H, w=W)
        else:
            raise RuntimeError(f'Direction {direction} not found.')

    def __repr__(self):
        scans = ', '.join(self.choices)
        return super().__repr__().replace(
            self.__class__.__name__, f'{self.__class__.__name__}[{scans}]')


class SelectiveScan(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx,
                u,
                delta,
                A,
                B,
                C,
                D=None,
                delta_bias=None,
                delta_softplus=False,
                nrows=1):
        assert nrows in [1, 2, 3, 4], f"{nrows}"  # 8+ is too slow to compile
        assert u.shape[1] % (B.shape[1] *
                             nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True

        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None,
                                                delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u,
            delta,
            A,
            B,
            C,
            D,
            None,
            delta_bias,
            dout,
            x,
            None,
            None,
            ctx.delta_softplus,
            False  # option to recompute out_z, not used here
        )

        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


class MultiScanVSSM(MultiScan):

    ALL_CHOICES = MultiScan.ALL_CHOICES

    def __init__(self, dim, choices=None):
        super().__init__(dim, choices=choices, token_size=None)

    def merge(self, xs):
        # xs: [B, K, D, L]
        # return: [B, D, L]
        # remove the padded tokens
        xs = [xs[:, i, :, :l] for i, l in enumerate(self.scan_lengths)]
        xs = super().multi_reverse(xs)
        xs = [x.transpose(-2, -1) for x in xs]
        x = super().forward(xs)
        return x

    def multi_scan(self, x):
        # x: [B, C, H, W]
        # return: [B, K, C, H * W]
        B, C, H, W = x.shape
        self.token_size = (H, W)

        xs = super().multi_scan(x)  # [[B, C, H, W], ...]
        self.scan_lengths = [x.shape[2] for x in xs]
        max_length = max(self.scan_lengths)

        # pad the tokens into the same length as VMamba compute all directions together
        new_xs = []
        for x in xs:
            if x.shape[2] < max_length:
                x = F.pad(x, (0, max_length - x.shape[2]))
            new_xs.append(x)
        return torch.stack(new_xs, 1)

    def __repr__(self):
        scans = ', '.join(self.choices)
        return super().__repr__().replace('MultiScanVSSM',
                                          f'MultiScanVSSM[{scans}]')


class SS2D(nn.Module):

    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3,  # < 2 means no conv
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        simple_init=False,
        directions=None,
        use_in_proj=True,
        use_out_proj=True,
        use_out_norm=True,
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = d_expand
        self.dt_rank = math.ceil(d_model /
                                 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(
            d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv

        self.use_in_proj = use_in_proj
        self.use_out_proj = use_out_proj
        self.use_out_norm = use_out_norm
        if self.use_out_norm:
            self.out_norm = nn.LayerNorm(d_inner)
        else:
            self.out_norm = nn.Identity()

        self.K = len(MultiScanVSSM.ALL_CHOICES) if directions is None else len(
            directions)
        self.K2 = self.K

        # in proj =======================================
        self.in_proj = nn.Linear(d_model,
                                 d_expand * 2,
                                 bias=bias,
                                 **factory_kwargs)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand,
                                     d_inner,
                                     kernel_size=1,
                                     bias=False,
                                     **factory_kwargs)
            self.out_rank = nn.Linear(d_inner,
                                      d_expand,
                                      bias=False,
                                      **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2),
                      bias=False,
                      **factory_kwargs) for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj],
                        dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min,
                         dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs],
                        dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state,
                                      d_inner,
                                      copies=self.K2,
                                      merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)  # (K * D)

        # out proj =======================================
        if self.use_out_proj:
            self.out_proj = nn.Linear(d_expand,
                                      d_model,
                                      bias=bias,
                                      **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.multi_scan = MultiScanVSSM(d_expand, choices=directions)

        if simple_init:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn(
                    (self.K2 * d_inner, self.d_state
                     )))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(
                torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

    @staticmethod
    def dt_init(dt_rank,
                d_inner,
                dt_scale=1.0,
                dt_init="random",
                dt_min=0.001,
                dt_max=0.1,
                dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) *
            (math.log(dt_max) - math.log(dt_min)) +
            math.log(dt_min)).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self,
                     x: torch.Tensor,
                     nrows=-1,
                     channel_first=False,
                     return_param=False):
        nrows = 1
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        if not return_param:
            x = multi_selective_scan(x,
                                     self.x_proj_weight,
                                     None,
                                     self.dt_projs_weight,
                                     self.dt_projs_bias,
                                     self.A_logs,
                                     self.Ds,
                                     self.out_norm,
                                     nrows=nrows,
                                     delta_softplus=True,
                                     multi_scan=self.multi_scan)
        else:
            x, params = multi_selective_scan(x,
                                             self.x_proj_weight,
                                             None,
                                             self.dt_projs_weight,
                                             self.dt_projs_bias,
                                             self.A_logs,
                                             self.Ds,
                                             self.out_norm,
                                             nrows=nrows,
                                             delta_softplus=True,
                                             multi_scan=self.multi_scan,
                                             return_param=True)
        if self.ssm_low_rank:
            x = self.out_rank(x)

        if not return_param:
            return x
        else:
            return x, params

    def forward(self, x: torch.Tensor, return_param=False):
        xz = self.in_proj(x)
        z_origin = xz.chunk(2, dim=-1)[1]
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
            z = self.act(z)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x))  # (b, d, h, w)
        else:
            xz = self.act(xz)
            x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
        if not return_param:
            y = self.forward_core(x,
                                  channel_first=(self.d_conv > 1),
                                  return_param=return_param)
        else:
            y, params = self.forward_core(x,
                                          channel_first=(self.d_conv > 1),
                                          return_param=return_param)

        y = y * z
        if self.use_out_proj:
            out = self.dropout(self.out_proj(y))
        else:
            out = self.dropout(y)
        if not return_param:
            return out, z_origin
        else:
            return out, [z_origin] + params
