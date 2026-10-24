from tinygrad import Tensor, nn, dtypes
from tinygrad.helpers import ContextVar, TRAINING
from math import sqrt
from typing import cast

FAKEQUANT = ContextVar("FAKEQUANT", 0)

# (weight_scale, act_scale_max, act_scale_min)
QUANTS = {"i4": (7, 7, -7),"i8": (127, 127, -127),"u4": (7, 15, 0),"u8": (127, 255, 0)}

class FakeQuantConv(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int = 1,
        padding: int | tuple[int, ...] | str = 0,
        dilation: int = 1,
        groups: int = 1,
        quant: str | None = None,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
        self.weight = Tensor.kaiming_uniform(*self.weight.shape, a=sqrt(5.0))
        self.is_quantized = quant is not None
        if self.is_quantized:
            self.weight_scale, self.act_scale_max, self.act_scale_min = QUANTS[quant]
            self.ema_max = Tensor.zeros(in_channels).is_param_(False)
            self.beta = 0.1
    def half_dot(self, x: Tensor, weight: Tensor) -> Tensor:
        return (x.cast(dtypes.half).conv2d(weight.cast(dtypes.half),self.bias, self.groups,self.stride, self.dilation,
                                           self.padding).cast(dtypes.float32))
    def __call__(self, x: Tensor) -> Tensor:
        if not self.is_quantized: return self.half_dot(x, self.weight)
        ws = (self.weight_scale / self.weight.detach().abs().max(axis=(1, 2, 3)).clamp(min_=1e-8))[:, None, None, None]
        w_fq = self.weight + ((self.weight * ws).clamp(-self.weight_scale, self.weight_scale).round() / ws - self.weight).detach()
        if not FAKEQUANT:
            if TRAINING:
                amax = x.abs().max(axis=(0, 2, 3))
                self.ema_max.assign((1 - self.beta) * self.ema_max+ self.beta * amax)
            return self.half_dot(x, w_fq)
        xs = (self.act_scale_max / self.ema_max.clamp(min_=1e-8))[None, :, None, None]  # [1, Cin, 1, 1]
        z = (x * xs).clamp(self.act_scale_min, self.act_scale_max)
        x_fq = (z + (z.round() - z).detach()) / xs
        return self.half_dot(x_fq, w_fq)

class FakeQuantLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, quant: str | None = None):
        super().__init__(in_features, out_features, bias=bias)
        self.is_quantized = quant is not None
        if self.is_quantized:
            self.weight_scale, self.act_scale_max, self.act_scale_min = QUANTS[quant]
            self.ema_max = Tensor.zeros(in_features).is_param_(False)
            self.beta = 0.1
    def half_dot(self, x: Tensor, weight: Tensor) -> Tensor:
        return x.cast(dtypes.half).linear(weight.cast(dtypes.half).transpose(), None).cast(dtypes.float32)
    def __call__(self, x: Tensor) -> Tensor:
        if not self.is_quantized: return self.half_dot(x, self.weight)
        ws = self.weight_scale / self.weight.detach().abs().max(axis=1, keepdim=True).clamp(min_=1e-8)
        w_fq = self.weight + ((self.weight * ws).clamp(-self.weight_scale, self.weight_scale).round() / ws - self.weight).detach()
        if not FAKEQUANT:
            if TRAINING:
                amax = x.detach().abs().reshape(-1, x.shape[-1]).max(axis=0)
                self.ema_max.assign((1 - self.beta) * self.ema_max+ self.beta * amax)
            return self.half_dot(x, w_fq)
        xs = (self.act_scale_max / self.ema_max.clamp(min_=1e-8))
        z = (x * xs).clamp(self.act_scale_min, self.act_scale_max)
        x_fq = (z + (z.round() - z).detach()) / xs
        return self.half_dot(x_fq, w_fq)

class ConvBlock:
    def __init__(self, cin, cout, quant):
        self.conv = FakeQuantConv(cin, cout, (3, 3), padding="same", quant=quant)
        self.norm = nn.BatchNorm2d(cout)
        self.cin, self.cout = cin, cout
        cast(Tensor, self.norm.weight).is_param_(False)
    def ema_tensors(self) -> list[Tensor]:
        return [self.conv.ema_max] if self.conv.is_quantized else []
    def compute_fwd_bwd_flops(self, bs: int, hw: int) -> int:
        fwd = 2 * bs * hw * hw * self.cin * 3 * 3 * self.cout
        return 3 * fwd
    def __call__(self, x: Tensor) -> Tensor:
        return self.norm(self.conv(x)).relu()

class ConvPoolBlock(ConvBlock):
    def __call__(self, x: Tensor) -> Tensor:
        return self.norm(self.conv(x)).max_pool2d(kernel_size=(2, 2)).relu()

class ResidualBlock:
    def __init__(self, c, quant):
        self.c1 = ConvBlock(c, c, quant)
        self.c2 = ConvBlock(c, c, quant)
    def ema_tensors(self) -> list[Tensor]:
        return self.c1.ema_tensors() + self.c2.ema_tensors()
    def compute_fwd_bwd_flops(self, bs: int, hw: int) -> int:
        return self.c1.compute_fwd_bwd_flops(bs, hw) + self.c2.compute_fwd_bwd_flops(bs, hw)
    def __call__(self, x: Tensor) -> Tensor:
        return x + self.c2(self.c1(x))

class ResNet:
    def __init__(self, channels: list[int] = [64, 128, 256, 512], cs: float = 1.0, res_idx={1, 3}, quantized: bool = False):
        self.quantized = quantized
        self.channels = [int(cs * c) for c in channels]
        self.layers = []
        ptr = 1
        cin, cnext = self.channels[ptr - 1], self.channels[ptr]
        self.prep = ConvBlock(3, cin, quant='i8' if quantized else None)
        while ptr < len(self.channels):
            cin, cnext = self.channels[ptr - 1], self.channels[ptr]
            self.layers.append(ConvPoolBlock(cin, cnext, quant='u4' if quantized else None))
            if ptr in res_idx:
                self.layers.append(ResidualBlock(cnext, quant='u4' if quantized else None))
            ptr += 1
        out_feats = int(self.channels[-1])
        self.classif = FakeQuantLinear(in_features=out_feats, out_features=10, bias=False, quant='u8' if quantized else None)
    def ema_tensors(self) -> list[Tensor]:
        ts = self.prep.ema_tensors()
        for layer in self.layers: ts += layer.ema_tensors()
        if self.quantized: ts.append(self.classif.ema_max)
        return ts
    def compute_costs(self, bs: int) -> tuple[float,float]:
        # hack but we just use 1/3 * compute_fwd_bwd_flops to get fwd flops
        M, Mprep = 4 if self.quantized else 16, 8 if self.quantized else 16
        hw = 32
        c_out = int(self.channels[-1])
        compute_cost = Mprep*self.prep.compute_fwd_bwd_flops(bs, hw)/3
        mem_cost = Mprep*sum(param.numel() for param in nn.state.get_parameters(self.prep))
        for layer in self.layers:
            compute_cost += M*layer.compute_fwd_bwd_flops(bs, hw)/3
            mem_cost += M*sum(param.numel() for param in nn.state.get_parameters(layer))
            if isinstance(layer, ConvPoolBlock):
                hw //= 2
        compute_cost += Mprep*2*bs*c_out*10
        mem_cost += Mprep*c_out*10
        return compute_cost, mem_cost
    def compute_fwd_bwd_flops(self, bs: int):
        hw = 32
        c_out = int(self.channels[-1])
        flops = self.prep.compute_fwd_bwd_flops(bs, hw)
        for layer in self.layers:
            flops += layer.compute_fwd_bwd_flops(bs, hw)
            if isinstance(layer, ConvPoolBlock):
                hw //= 2
        classif = 2 * bs * c_out * 10
        flops += 3 * classif
        return flops
    def __call__(self, x: Tensor) -> Tensor:
        x = self.prep(x)
        for layer in self.layers:
            x = layer(x)
        x = x.max_pool2d(kernel_size=(4, 4)).flatten(1)
        return self.classif(x) / 8.0
