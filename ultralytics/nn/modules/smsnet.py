from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv
from .head import Detect


def _channel_shuffle(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.reshape(b * c // 2, 2, h * w).permute(1, 0, 2).reshape(2, -1, c // 2, h, w)
    return torch.cat((x[0], x[1]), 1)


def _normal_init(module: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, 0.0)


class HSigmoid(nn.Module):
    def __init__(self, inplace: bool = True, bias: float = 3.0, divisor: float = 6.0):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.bias = bias
        self.divisor = divisor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.bias) / self.divisor


def build_activation_layer(cfg: dict | None) -> nn.Module:
    cfg = cfg or {"type": "HSigmoid", "bias": 3.0, "divisor": 6.0}
    if cfg.get("type") != "HSigmoid":
        raise ValueError(f"Unsupported activation type: {cfg.get('type')}")
    return HSigmoid(bias=cfg.get("bias", 3.0), divisor=cfg.get("divisor", 6.0))


class DyReLU(nn.Module):
    def __init__(
        self,
        inp: int,
        reduction: int = 4,
        lambda_a: float = 1.0,
        k2: bool = True,
        use_bias: bool = True,
        use_spatial: bool = False,
        init_a: tuple[float, float] = (1.0, 0.0),
        init_b: tuple[float, float] = (0.0, 0.0),
    ):
        super().__init__()
        self.oup = inp
        self.lambda_a = lambda_a * 2
        self.k2 = k2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.use_bias = use_bias
        self.exp = 4 if k2 and use_bias else 2 if (k2 or use_bias) else 1
        self.init_a = init_a
        self.init_b = init_b
        squeeze = max(inp // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(inp, squeeze),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze, self.oup * self.exp),
            HSigmoid(),
        )
        self.spa = (
            nn.Sequential(
                nn.Conv2d(inp, 1, kernel_size=1),
                nn.BatchNorm2d(1),
            )
            if use_spatial
            else None
        )

    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if isinstance(x, list):
            x_in, x_out = x
        else:
            x_in = x_out = x
        b, c, h, w = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, self.oup * self.exp, 1, 1)

        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
            a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
            b1 = b1 - 0.5 + self.init_b[0]
            b2 = b2 - 0.5 + self.init_b[1]
            out = torch.max(x_out * a1 + b1, x_out * a2 + b2)
        elif self.exp == 2:
            if self.use_bias:
                a1, b1 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
                b1 = b1 - 0.5 + self.init_b[0]
                out = x_out * a1 + b1
            else:
                a1, a2 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
                a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
                out = torch.max(x_out * a1, x_out * a2)
        else:
            a1 = (y - 0.5) * self.lambda_a + self.init_a[0]
            out = x_out * a1

        if self.spa is not None:
            ys = self.spa(x_in).view(b, -1)
            ys = F.softmax(ys, dim=1).view(b, 1, h, w) * h * w
            ys = F.hardtanh(ys, 0, 3, inplace=True) / 3
            out = out * ys
        return out


class SlimGSConv(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p=None, g: int = 1, d: int = 1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, Conv.default_act)
        self.cv2 = Conv(c_, c_, 5, 1, p, c_, d, Conv.default_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.cv1(x)
        return _channel_shuffle(torch.cat((x1, self.cv2(x1)), 1))


class BNConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, bias=False):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.seq(x))


class MSBlock(nn.Module):
    def __init__(self, in_channels: int, shared_channels: int):
        super().__init__()
        self.shared_conv = BNConv2d(in_channels, shared_channels, 1, 1, 0, bias=False)
        self.branch1 = nn.MaxPool2d(3, 1, 1)
        self.branch2 = BNConv2d(shared_channels, shared_channels, 3, 1, 1, bias=False)
        self.branch3 = nn.Sequential(
            BNConv2d(shared_channels, shared_channels, 3, 1, 1, bias=False),
            BNConv2d(shared_channels, shared_channels, 3, 1, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shared = self.shared_conv(x)
        out = torch.cat((self.branch1(x_shared), self.branch2(x_shared), self.branch3(x_shared), x_shared), 1)
        return _channel_shuffle(out)


class CSSPD(nn.Module):
    def __init__(self, inc: int, ouc: int):
        super().__init__()
        self.conv_1 = Conv(ouc, ouc // 4, k=1)
        self.conv3 = Conv(inc, ouc, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3(x)
        patches = (x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2])
        x = torch.cat([self.conv_1(p) for p in patches], 1)
        return _channel_shuffle(x)


class MSSPD(nn.Module):
    def __init__(self, inc: int, ouc: int):
        super().__init__()
        self.conv_1 = Conv(ouc, ouc // 4, k=1)
        self.inception = MSBlock(inc, ouc // 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inception(x)
        patches = (x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2])
        x = torch.cat([self.conv_1(p) for p in patches], 1)
        return _channel_shuffle(x)


class MBFusionA(nn.Module):
    def __init__(self, in_channel_list: list[int], out_channels: int):
        super().__init__()
        self.cv1 = (
            Conv(in_channel_list[0], out_channels, act=nn.ReLU())
            if in_channel_list[0] != out_channels
            else nn.Identity()
        )
        self.cv2 = (
            Conv(in_channel_list[1], out_channels, act=nn.ReLU())
            if in_channel_list[1] != out_channels
            else nn.Identity()
        )
        self.cv3 = (
            Conv(in_channel_list[2], out_channels, act=nn.ReLU())
            if in_channel_list[2] != out_channels
            else nn.Identity()
        )
        self.cv_fuse = Conv(out_channels * 3, out_channels, act=nn.ReLU())
        self.downsample_avg = nn.AvgPool2d(kernel_size=2, stride=2)
        self.downsample_max = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        _, _, h, w = x[1].shape
        x0 = self.cv1(0.5 * self.downsample_avg(x[0]) + 0.5 * self.downsample_max(x[0]))
        x1 = self.cv2(x[1])
        x2 = self.cv3(F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=False))
        return self.cv_fuse(torch.cat((x0, x1, x2), dim=1))


class MBFusionB(nn.Module):
    def __init__(self, in_channel_list: list[int], out_channels: int):
        super().__init__()
        self.cv1 = (
            Conv(in_channel_list[0], out_channels, act=nn.ReLU())
            if in_channel_list[0] != out_channels
            else nn.Identity()
        )
        self.cv2 = (
            Conv(in_channel_list[1], out_channels, act=nn.ReLU())
            if in_channel_list[1] != out_channels
            else nn.Identity()
        )
        self.cv3 = (
            Conv(in_channel_list[2], out_channels, act=nn.ReLU())
            if in_channel_list[2] != out_channels
            else nn.Identity()
        )
        self.cv_fuse = Conv(out_channels * 3, out_channels, act=nn.ReLU())

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        _, _, h, w = x[1].shape
        x0 = self.cv1(F.interpolate(x[0], size=(h, w), mode="bilinear", align_corners=False))
        x1 = self.cv2(x[1])
        x2 = self.cv3(F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=False))
        return self.cv_fuse(torch.cat((x0, x1, x2), dim=1))


class MBFusionC(nn.Module):
    def __init__(self, in_channel_list: list[int], out_channels: int):
        super().__init__()
        self.cv1 = (
            Conv(in_channel_list[0], out_channels, act=nn.ReLU())
            if in_channel_list[0] != out_channels
            else nn.Identity()
        )
        self.cv2 = (
            Conv(in_channel_list[1], out_channels, act=nn.ReLU())
            if in_channel_list[1] != out_channels
            else nn.Identity()
        )
        self.cv_fuse = Conv(out_channels * 2, out_channels, act=nn.ReLU())

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        _, _, h, w = x[1].shape
        x0 = self.cv1(F.interpolate(x[0], size=(h, w), mode="bilinear", align_corners=False))
        x1 = self.cv2(x[1])
        return self.cv_fuse(torch.cat((x0, x1), dim=1))


class LightDyHeadBlockM(nn.Module):
    def __init__(self, low_channels: int, high_channels: int, act_cfg: dict | None = None):
        super().__init__()
        act_cfg = act_cfg or {"type": "HSigmoid", "bias": 3.0, "divisor": 6.0}
        self.scale_attn_module_0 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(low_channels, 1, 1),
            nn.ReLU(inplace=True),
            build_activation_layer(act_cfg),
        )
        self.task_attn_module_0 = DyReLU(low_channels)
        self.conv_1 = Conv(high_channels, low_channels, 1)

        self.scale_attn_module_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, 1, 1),
            nn.ReLU(inplace=True),
            build_activation_layer(act_cfg),
        )
        self.task_attn_module_1 = DyReLU(high_channels)
        self.conv_2 = SlimGSConv(low_channels, high_channels, 3, 2)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                _normal_init(module, 0.0, 0.01)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        outs = []
        x = list(x)
        for level in range(len(x)):
            if level > 0:
                sum_feat = x[level] * self.scale_attn_module_1(x[level])
                summed_levels = 1
                x[level - 1] = self.conv_2(x[level - 1])
                sum_feat += x[level - 1] * self.scale_attn_module_1(x[level - 1])
                summed_levels += 1
                outs.append(self.task_attn_module_1(sum_feat / summed_levels))
            if level < len(x) - 1:
                sum_feat = x[level] * self.scale_attn_module_0(x[level])
                summed_levels = 1
                high_feat = self.conv_1(x[level + 1])
                high_feat = F.interpolate(high_feat, size=x[level].shape[-2:], mode="bilinear", align_corners=False)
                sum_feat += high_feat * self.scale_attn_module_0(high_feat)
                summed_levels += 1
                outs.append(self.task_attn_module_0(sum_feat / summed_levels))
        return outs


class STHead(Detect):
    def __init__(self, nc: int = 80, block_num: int = 1, reg_max: int = 16, end2end: bool = False, ch: tuple = ()):
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        if len(ch) != 2:
            raise ValueError(f"STHead expects exactly 2 input feature levels, but got {len(ch)}")
        self.dyhead = nn.Sequential(*[LightDyHeadBlockM(ch[0], ch[1]) for _ in range(block_num)])
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)
        self.cv2 = nn.ModuleList(
            nn.Sequential(SlimGSConv(x, c2, 3), SlimGSConv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        return super().forward_head(self.dyhead(list(x)), box_head, cls_head)

    def bias_init(self):
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):
                a[-1].bias.data[:] = 1.0
                b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)


__all__ = ("CSSPD", "MSSPD", "MBFusionA", "MBFusionB", "MBFusionC", "STHead")
