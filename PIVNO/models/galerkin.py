import torch
import torch.nn as nn
from functools import partial
from . import register

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias
        return out


@register('galerkin')
class simple_attn(nn.Module):
    #midc=128  heads=16
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads  #8
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv2d(midc, 3*midc, 1)
        self.o_proj1 = nn.Conv2d(midc, midc, 1)
        self.o_proj2 = nn.Conv2d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = nn.GELU()

        self.dwconv = nn.Conv2d(midc, midc, 3, 1, 1, bias=True, groups=midc)

        self.compute_cis = partial(self.compute_mixed_cis, num_heads=self.heads)
        freqs = self.init_2d_freqs(
            dim=self.headc, num_heads=self.heads, theta=100,
            rotate=True
        ).view(2, -1)
        self.freqs = nn.Parameter(freqs, requires_grad=True)

    def init_t_xy(self, end_x: int, end_y: int):
        t = torch.arange(end_x * end_y, dtype=torch.float32)
        t_x = (t % end_x).float()
        t_y = torch.div(t, end_x, rounding_mode='floor').float()
        return t_x, t_y

    def reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
            shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
        elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
            shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    def apply_rotary_emb(self, xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2).contiguous())
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2).contiguous())
        freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

    def compute_mixed_cis(self, freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
        N = t_x.shape[0]
        # No float 16 for this range
        with torch.cuda.amp.autocast(enabled=False):
            freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
            freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
            freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
        return freqs_cis

    def init_2d_freqs(self, dim: int, num_heads: int, theta: float = 100.0, rotate: bool = True):
        freqs_x = []
        freqs_y = []
        mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
        for i in range(num_heads):
            angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
            fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi / 2 + angles)], dim=-1)
            fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi / 2 + angles)], dim=-1)
            freqs_x.append(fx)
            freqs_y.append(fy)
        freqs_x = torch.stack(freqs_x, dim=0)
        freqs_y = torch.stack(freqs_y, dim=0)
        freqs = torch.stack([freqs_x, freqs_y], dim=0)
        return freqs

    def forward(self, x, name='0'):
        B, C, H, W = x.shape
        bias = x

        qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H*W, self.heads, 3*self.headc)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        #ROPE
        # q(B,heads,n,headc)  (b,16,4096,8)
        t_x, t_y = self.init_t_xy(end_x=W, end_y=H)
        t_x, t_y = t_x.to(x.device), t_y.to(x.device)
        freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
        k, v = self.apply_rotary_emb(k, v, freqs_cis=freqs_cis)

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2,-1), v) / (H*W)

        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)

        ret = v.permute(0, 3, 1, 2) + bias
        z = self.o_proj1(ret)
        z = self.dwconv(z)
        bias = self.o_proj2(self.act(z)) + bias
        
        return bias



