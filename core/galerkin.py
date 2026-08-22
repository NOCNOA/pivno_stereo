import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

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

class simple_attn(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv2d(midc, 3*midc, 1)
        self.o_proj1 = nn.Conv2d(midc, midc, 1)
        self.o_proj2 = nn.Conv2d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = GELU()
    
    def forward(self, x, name='0'):
        B, C, H, W = x.shape
        bias = x

        qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H*W, self.heads, 3*self.headc) # B, H*W, heads, 3*headc
        qkv = qkv.permute(0, 2, 1, 3) # B, heads, H*W, 3*headc
        q, k, v = qkv.chunk(3, dim=-1) # B, heads, H*W, headc

        k = self.kln(k) # B, heads, H*W, headc
        v = self.vln(v) # B, heads, H*W, headc

        
        v = torch.matmul(k.transpose(-2,-1), v / (H*W)) # B, heads, H*W, H*W

        if torch.isnan(v).any() or torch.isnan(v).any():
            print('v, isnan-isinf_v', torch.isnan(x).any(), torch.isnan(q).any(), torch.isnan(k).any(), torch.isnan(v).any(), flush=True)
        
        v = torch.matmul(q, v)

        if torch.isnan(v).any() or torch.isnan(v).any():
            print('v_, isnan-isinf_v_', torch.isnan(q).any(), torch.isnan(k).any(), torch.isnan(v).any(), flush=True)

        v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)

        ret = v.permute(0, 3, 1, 2) + bias
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        
        return bias


class simple_attn_rope_2d(nn.Module):
    def __init__(self, midc, heads, theta=100.0):
        super().__init__()
        if midc % heads != 0:
            raise ValueError(f"midc={midc} must be divisible by heads={heads}")

        self.headc = midc // heads
        if self.headc % 2 != 0:
            raise ValueError(f"head dimension must be even for RoPE, got {self.headc}")

        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv2d(midc, 3 * midc, 1)
        self.o_proj1 = nn.Conv2d(midc, midc, 1)
        self.o_proj2 = nn.Conv2d(midc, midc, 1)
        self.dwconv = nn.Conv2d(midc, midc, 3, 1, 1, bias=True, groups=midc)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))
        self.act = GELU()

        freqs = self.init_2d_freqs(
            dim=self.headc, num_heads=self.heads, theta=theta, rotate=True
        ).view(2, -1)
        self.freqs = nn.Parameter(freqs, requires_grad=True)

    @staticmethod
    def init_t_xy(end_x, end_y):
        t = torch.arange(end_x * end_y, dtype=torch.float32)
        t_x = (t % end_x).float()
        t_y = torch.div(t, end_x, rounding_mode='floor').float()
        return t_x, t_y

    @staticmethod
    def reshape_for_broadcast(freqs_cis, x):
        ndim = x.ndim
        if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
            shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
        elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
            shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
        else:
            raise ValueError(f"Unexpected freqs_cis shape {freqs_cis.shape} for x shape {x.shape}")
        return freqs_cis.view(*shape)

    def apply_rotary_emb(self, xq, xk, freqs_cis):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2).contiguous())
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2).contiguous())
        freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def compute_mixed_cis(self, freqs, t_x, t_y, num_heads):
        n = t_x.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(n, num_heads, -1).permute(1, 0, 2)
            freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(n, num_heads, -1).permute(1, 0, 2)
            freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
        return freqs_cis

    @staticmethod
    def init_2d_freqs(dim, num_heads, theta=100.0, rotate=True):
        freqs_x = []
        freqs_y = []
        mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
        for _ in range(num_heads):
            angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
            fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi / 2 + angles)], dim=-1)
            fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi / 2 + angles)], dim=-1)
            freqs_x.append(fx)
            freqs_y.append(fy)
        freqs_x = torch.stack(freqs_x, dim=0)
        freqs_y = torch.stack(freqs_y, dim=0)
        return torch.stack([freqs_x, freqs_y], dim=0)

    def forward(self, x, name='0'):
        del name
        b, c, h, w = x.shape
        bias = x

        qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(b, h * w, self.heads, 3 * self.headc)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        t_x, t_y = self.init_t_xy(end_x=w, end_y=h)
        t_x = t_x.to(x.device)
        t_y = t_y.to(x.device)
        freqs_cis = self.compute_mixed_cis(self.freqs, t_x, t_y, self.heads)
        q, k = self.apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2, -1), v) / (h * w)
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(b, h, w, c)

        ret = v.permute(0, 3, 1, 2) + bias
        z = self.o_proj1(ret)
        z = self.dwconv(z)
        return self.o_proj2(self.act(z)) + bias


class simple_cross_attn_rope_2d(simple_attn_rope_2d):
    """Linear RoPE cross attention with left queries and right key/value."""

    def __init__(self, midc, heads, theta=100.0):
        super().__init__(midc, heads, theta)
        del self.qkv_proj
        self.q_proj = nn.Conv2d(midc, midc, 1)
        self.kv_proj = nn.Conv2d(midc, 2 * midc, 1)

    def forward(self, query, key_value):
        if query.shape != key_value.shape:
            raise ValueError(
                f"Cross-RoPE requires equal query/key-value shapes, got "
                f"{tuple(query.shape)} and {tuple(key_value.shape)}"
            )
        b, c, h, w = query.shape
        bias = query
        q = self.q_proj(query).permute(0, 2, 3, 1).reshape(b, h * w, self.heads, self.headc)
        kv = self.kv_proj(key_value).permute(0, 2, 3, 1).reshape(b, h * w, self.heads, 2 * self.headc)
        q = q.permute(0, 2, 1, 3)
        k, v = kv.permute(0, 2, 1, 3).chunk(2, dim=-1)

        t_x, t_y = self.init_t_xy(end_x=w, end_y=h)
        freqs_cis = self.compute_mixed_cis(
            self.freqs, t_x.to(query.device), t_y.to(query.device), self.heads
        )
        q, k = self.apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        k, v = self.kln(k), self.vln(v)
        context = torch.matmul(k.transpose(-2, -1), v) / (h * w)
        out = torch.matmul(q, context).permute(0, 2, 1, 3).reshape(b, h, w, c)

        ret = out.permute(0, 3, 1, 2) + bias
        z = self.dwconv(self.o_proj1(ret))
        return self.o_proj2(self.act(z)) + bias

class simple_attn_3d(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv3d(midc, 3*midc, 1)
        self.o_proj1 = nn.Conv3d(midc, midc, 1)
        self.o_proj2 = nn.Conv3d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = GELU()
    
    def forward(self, x, name='0'):
        B, C, D, H, W = x.shape
        bias = x

        # print(f'galerkin before {x.shape}', flush=True)
        
        qkv = self.qkv_proj(x)
        qkv = qkv.permute(0, 2, 3, 4, 1)
        qkv = qkv.reshape(B*D, H*W, self.heads, 3*self.headc)

        # print(f'galerkin after {qkv.shape}', flush=True)

        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1) # B*D, heads, H*W, headc

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2,-1), v/(H*W))
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, D, H, W, C)

        ret = v.permute(0, 4, 1, 2, 3) + bias
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        
        return bias
    
class simple_attn_3d2(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv3d(midc, 3*midc, 1)
        self.o_proj1 = nn.Conv3d(midc, midc, 1)
        self.o_proj2 = nn.Conv3d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = GELU()
    
    def forward(self, x, name='0'):
        B, C, D, H, W = x.shape
        bias = x

        # print(f'galerkin before {x.shape}', flush=True)
        
        qkv = self.qkv_proj(x)
        qkv = qkv.permute(0, 3, 4, 2, 1)  #B H W D C
        qkv = qkv.reshape(B*H*W, D, self.heads, 3*self.headc)

        # print(f'galerkin after {qkv.shape}', flush=True)

        qkv = qkv.permute(0, 2, 1, 3)  #B*H*W, self.heads, D, 3*self.headc
        q, k, v = qkv.chunk(3, dim=-1) # B*H*W, heads, D, headc

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2,-1), v/(D))
        v = torch.matmul(q, v)   # B*H*W, heads, D, headc
        v = v.permute(0, 2, 1, 3).reshape(B, H, W, D, C)# B*H*W, heads, D, headc  ->  B*H*W, D, heads, headc

        ret = v.permute(0, 4, 3, 1, 2) + bias
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        
        return bias

class simple_attn_3d3(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv3d(midc, 3*midc, 1)
        self.o_proj1 = nn.Conv3d(midc, midc, 1)
        self.o_proj2 = nn.Conv3d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = GELU()
    
    def forward(self, x, name='0'):
        B, C, D, H, W = x.shape
        bias = x
        
        # print(f'galerkin before {x.shape}', flush=True)
        
        qkv = self.qkv_proj(x)
        qkv = qkv.permute(0, 2, 3, 4, 1).contiguous()
        qkv = qkv.reshape(B*D*H, W, self.heads, 3*self.headc)

        # print(f'galerkin after {qkv.shape}', flush=True)

        qkv = qkv.permute(0, 2, 1, 3).contiguous()  #B*D*H, self.heads, W, 3*self.headc
        q, k, v = qkv.chunk(3, dim=-1) #H*B*D, heads, W, headc

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2,-1), v/(W)) #H*B*D, heads, headc, headc
        v = torch.matmul(q, v)  #B*D*H, heads, W, headc
        v = v.permute(0, 2, 1, 3).contiguous().reshape(B, D, H, W, C) #B, D, H, W, C

        ret = v.permute(0, 4, 1, 2, 3).contiguous()
        ret = ret + bias #B, C, D, H, W
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias 
        
        return bias

class simple_attn_3d4(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv3d(3*midc, 3*midc, 1)
        self.o_proj1 = nn.Conv3d(midc, midc, 1)
        self.o_proj2 = nn.Conv3d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = GELU()
    
    def forward(self, x, name='0'):
        B, C, D, H, W = x.shape
        bias = x
        x = torch.cat([x, x, x], dim=1)
        # print(f'galerkin before {x.shape}', flush=True)
        
        qkv = self.qkv_proj(x)
        qkv = qkv.permute(0, 2, 4, 3, 1)#BDWHC
        qkv = qkv.reshape(B*D*W, H, self.heads, 3*self.headc)

        # print(f'galerkin after {qkv.shape}', flush=True)

        qkv = qkv.permute(0, 2, 1, 3)  #B*D*W, self.heads, H, 3*self.headc
        q, k, v = qkv.chunk(3, dim=-1) #W*B*D, heads, H, headc

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2,-1), v/(H)) #W*B*D, heads, headc, headc
        v = torch.matmul(q, v)  #B*D*W, heads, H, headc
        v = v.permute(0, 2, 1, 3).reshape(B, D, W, H, C) #B, D, W, H, C

        ret = v.permute(0, 4, 1, 3, 2) + bias #B, C, D, H, W
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias 
        
        return bias

class simple_attn_3d5(nn.Module):
    def __init__(self, midc, featc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc
        self.featc = featc
        self.kv_proj1 = nn.Conv2d(featc, midc, 1)
        self.kv_proj2 = nn.Conv3d(midc, 2 * midc, 1)

        self.q_proj = nn.Conv3d(midc, midc, 1)
        self.o_proj1 = nn.Conv3d(midc, midc, 1)
        self.o_proj2 = nn.Conv3d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = GELU()
    
    def forward(self, x, feat, name='0'):
        B, C, D, H, W = x.shape
        bias = x
        
        # print(f'galerkin before {x.shape}', flush=True)
        
        q = self.q_proj(x)
        feat = self.kv_proj1(feat)[:, :, D, :, :]
        feat = self.kv_proj2(feat)#B 2C D H W
        kv = feat.permute(0, 2, 3, 4, 1)
        kv = kv.reshape(B*D*H, W, self.heads, 2*self.headc)
        q  = q.reshape(B*D*H, W, self.heads, self.headc)
        # print(f'galerkin after {qkv.shape}', flush=True)
        q = q.permute(0, 2, 1, 3)  #B*D*H, self.heads, W, self.headc
        kv = kv.permute(0, 2, 1, 3)  #B*D*H, self.heads, W, 2*self.headc
        k, v = kv.chunk(2, dim=-1) #H*B*D, heads, W, headc

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2,-1), v/(W)) #H*B*D, heads, headc, headc
        v = torch.matmul(q, v)  #B*D*H, heads, W, headc
        v = v.permute(0, 2, 1, 3).reshape(B, D, H, W, C) #B, D, H, W, C

        ret = v.permute(0, 4, 1, 2, 3) + bias #B, C, D, H, W
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias 
        
        return bias


class simple_attn_3d_hwd_patch(nn.Module):
    """
    Fast HWD patch Galerkin attention for non-overlapping 3D patches.

    Input : x [B, C, D, H, W]
    Output: y [B, C, D, H, W]

    Constraints:
    - stride == patch
    - D % pd == 0, H % ph == 0, W % pw == 0
    """

    def __init__(self, midc: int, heads: int, patch=(4, 4, 4)):
        super().__init__()
        assert midc % heads == 0, "midc must be divisible by heads"

        self.midc = midc
        self.heads = heads
        self.headc = midc // heads

        if isinstance(patch, int):
            self.pd = self.ph = self.pw = int(patch)
        else:
            self.pd, self.ph, self.pw = map(int, patch)

        self.qkv_proj = nn.Conv3d(midc, 3 * midc, kernel_size=1, bias=True)
        self.o_proj1 = nn.Conv3d(midc, midc, kernel_size=1, bias=True)
        self.o_proj2 = nn.Conv3d(midc, midc, kernel_size=1, bias=True)

        self.kln = nn.LayerNorm(self.headc, elementwise_affine=True)
        self.vln = nn.LayerNorm(self.headc, elementwise_affine=True)

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, name: str = "0"):
        if x.ndim != 5:
            raise ValueError(f"Expected [B,C,D,H,W], got {tuple(x.shape)}")

        B, C, D, H, W = x.shape
        bias = x

        pd, ph, pw = self.pd, self.ph, self.pw
        assert D % pd == 0, f"D={D} must be divisible by pd={pd}"
        assert H % ph == 0, f"H={H} must be divisible by ph={ph}"
        assert W % pw == 0, f"W={W} must be divisible by pw={pw}"

        Nd = D // pd
        Nh = H // ph
        Nw = W // pw
        Np = Nd * Nh * Nw
        L = pd * ph * pw

        qkv = self.qkv_proj(x)  # [B, 3C, D, H, W]

        # [B, 3C, Nd, pd, Nh, ph, Nw, pw]
        qkv = qkv.view(B, 3 * C, Nd, pd, Nh, ph, Nw, pw)

        # -> [B, Nd, Nh, Nw, pd, ph, pw, 3C]
        qkv = qkv.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()

        # -> [B, Np, L, 3C]
        qkv = qkv.view(B, Np, L, 3 * C)

        # -> [B*Np, heads, L, 3*headc]
        qkv = qkv.view(B * Np, L, self.heads, 3 * self.headc).permute(0, 2, 1, 3).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)  # [B*Np, heads, L, headc]

        k = self.kln(k)
        v = self.vln(v)

        # Galerkin
        A = torch.matmul(k.transpose(-2, -1), v / float(L))   # [B*Np, heads, headc, headc]
        y = torch.matmul(q, A)                                 # [B*Np, heads, L, headc]

        # -> [B, Np, L, C]
        y = y.permute(0, 2, 1, 3).contiguous().view(B, Np, L, C)

        # restore patches
        # [B, Nd, Nh, Nw, pd, ph, pw, C]
        y = y.view(B, Nd, Nh, Nw, pd, ph, pw, C)

        # -> [B, C, Nd, pd, Nh, ph, Nw, pw]
        y = y.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()

        # -> [B, C, D, H, W]
        y = y.view(B, C, D, H, W)

        ret = bias + y
        out = ret + self.o_proj2(self.act(self.o_proj1(ret)))
        return out

class simple_attn_1d(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv1d(midc, 3*midc, 1)
        self.o_proj1 = nn.Conv1d(midc, midc, 1)
        self.o_proj2 = nn.Conv1d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = GELU()
    
    def forward(self, x, name='0'):
        B, C, W = x.shape
        bias = x

        qkv = self.qkv_proj(x).permute(0, 2, 1).reshape(B, W, self.heads, 3*self.headc)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2,-1), v) / (W)
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, W, C)

        ret = v.permute(0, 2, 1) + bias
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        
        return bias

class FlashMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None, window_size=(-1,-1)):
        """
        @query: (B,L,C)
        """
        B,L,C = query.shape
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        Q = Q.view(Q.size(0), Q.size(1), self.num_heads, self.head_dim)
        K = K.view(K.size(0), K.size(1), self.num_heads, self.head_dim)
        V = V.view(V.size(0), V.size(1), self.num_heads, self.head_dim)

        attn_output = F.scaled_dot_product_attention(Q, K, V)

        attn_output = attn_output.reshape(B,L,-1)
        output = self.out_proj(attn_output)

        return output
        
class FlashAttentionTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1, act=nn.GELU, norm=nn.LayerNorm):
        super().__init__()
        self.self_attn = FlashMultiheadAttention(embed_dim, num_heads)
        self.act = act()

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = norm(embed_dim)
        self.norm2 = norm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, window_size=(-1, -1)):
        src2 = self.self_attn(src, src, src, src_mask, window_size=window_size)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class FlashAttentionTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1, act=nn.GELU, norm=nn.LayerNorm):
        super().__init__()
        self.self_attn = FlashMultiheadAttention(embed_dim, num_heads)
        self.act = act()

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = norm(embed_dim)
        self.norm2 = norm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, window_size=(-1, -1)):
        src2 = self.self_attn(src, src, src, src_mask, window_size=window_size)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src  

class PositionalEmbedding(nn.Module):
  def __init__(self, d_model, max_len=512):
    super().__init__()

    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model).float()
    pe.require_grad = False

    position = torch.arange(0, max_len).float().unsqueeze(1)  #(N,1)
    div_term = (torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).exp()[None]

    pe[:, 0::2] = torch.sin(position * div_term)  #(N, d_model/2)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)
    self.pe = pe
    # self.register_buffer('pe', pe)  #(1, max_len, D)


  def forward(self, x, resize_embed=False):
    '''
    @x: (B,N,D)
    '''
    self.pe = self.pe.to(x.device).to(x.dtype)
    pe = self.pe
    if pe.shape[1]<x.shape[1]:
      if resize_embed:
        pe = F.interpolate(pe.permute(0,2,1), size=x.shape[1], mode='linear', align_corners=False).permute(0,2,1)
      else:
        raise RuntimeError(f'x:{x.shape}, pe:{pe.shape}')
    return x + pe[:, :x.size(1)]

class CostVolumeDisparityAttention(nn.Module):
  def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, act=nn.GELU, norm_first=False, num_transformer=6, max_len=512, resize_embed=False):
    super().__init__()
    self.resize_embed = resize_embed
    self.sa = nn.ModuleList([])
    for _ in range(num_transformer):
      self.sa.append(FlashAttentionTransformerEncoderLayer(embed_dim=d_model, num_heads=nhead, dim_feedforward=dim_feedforward, act=act, dropout=dropout))
    self.pos_embed0 = PositionalEmbedding(d_model, max_len=max_len)


  def forward(self, cv, window_size=(-1,-1)):
    """
    @cv: (B,C,D,H,W) where D is max disparity
    """
    x = cv
    B,C,D,H,W = x.shape
    x = x.permute(0,3,4,2,1).reshape(B*H*W, D, C)
    x = self.pos_embed0(x, resize_embed=self.resize_embed)  #!NOTE No resize since disparity is pre-determined
    for i in range(len(self.sa)):
        x = self.sa[i](x, window_size=window_size)
    x = x.reshape(B,H,W,D,C).permute(0,4,3,1,2)

    return x

import torch
import torch.nn as nn


class simple_attn_3d_hw_patch(nn.Module):
    """
    Each disparity/depth slice d is processed independently.
    Only H, W are patchified (non-overlapping patches).

    Input : x [B, C, D, H, W]
    Output: y [B, C, D, H, W]

    Constraints:
    - H % ph == 0
    - W % pw == 0
    """

    def __init__(self, midc, heads, patch=(4, 4)):
        super().__init__()
        assert midc % heads == 0, "midc must be divisible by heads"

        self.midc = midc
        self.heads = heads
        self.headc = midc // heads

        if isinstance(patch, int):
            self.ph = self.pw = int(patch)
        else:
            self.ph, self.pw = map(int, patch)

        self.qkv_proj = nn.Conv3d(midc, 3 * midc, kernel_size=1, bias=True)
        self.o_proj1 = nn.Conv3d(midc, midc, kernel_size=1, bias=True)
        self.o_proj2 = nn.Conv3d(midc, midc, kernel_size=1, bias=True)

        # 如果你项目里有自定义 LayerNorm，也可以替换成：
        # self.kln = LayerNorm((self.heads, 1, self.headc))
        # self.vln = LayerNorm((self.heads, 1, self.headc))
        self.kln = nn.LayerNorm(self.headc, elementwise_affine=True)
        self.vln = nn.LayerNorm(self.headc, elementwise_affine=True)

        self.act = nn.GELU()

    def forward(self, x, name='0'):
        if x.ndim != 5:
            raise ValueError(f"Expected [B,C,D,H,W], got {tuple(x.shape)}")

        B, C, D, H, W = x.shape

        ph, pw = self.ph, self.pw
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        bias = x
        _, _, _, Hp, Wp = x.shape

        Nh = Hp // ph
        Nw = Wp // pw
        Np = Nh * Nw
        L = ph * pw

        # [B, 3C, D, H, W]
        qkv = self.qkv_proj(x)

        # [B, 3C, D, Nh, ph, Nw, pw]
        qkv = qkv.view(B, 3 * C, D, Nh, ph, Nw, pw)

        # [B, D, Nh, Nw, ph, pw, 3C]
        qkv = qkv.permute(0, 2, 3, 5, 4, 6, 1).contiguous()

        # [B, D, Np, L, 3C]
        qkv = qkv.view(B, D, Np, L, 3 * C)

        # [B*D*Np, L, heads, 3*headc]
        qkv = qkv.view(B * D * Np, L, self.heads, 3 * self.headc)

        # [B*D*Np, heads, L, 3*headc]
        qkv = qkv.permute(0, 2, 1, 3).contiguous()

        # q, k, v: [B*D*Np, heads, L, headc]
        q, k, v = qkv.chunk(3, dim=-1)

        k = self.kln(k)
        v = self.vln(v)

        # Galerkin attention
        # [B*D*Np, heads, headc, headc]
        A = torch.matmul(k.transpose(-2, -1), v / float(L))

        # [B*D*Np, heads, L, headc]
        y = torch.matmul(q, A)

        # [B*D*Np, L, heads, headc]
        y = y.permute(0, 2, 1, 3).contiguous()

        # [B, D, Np, L, C]
        y = y.view(B, D, Np, L, C)

        # [B, D, Nh, Nw, ph, pw, C]
        y = y.view(B, D, Nh, Nw, ph, pw, C)

        # [B, C, D, Nh, ph, Nw, pw]
        y = y.permute(0, 6, 1, 2, 4, 3, 5).contiguous()

        # [B, C, D, H, W]
        y = y.view(B, C, D, Hp, Wp)

        ret = bias + y
        out = ret + self.o_proj2(self.act(self.o_proj1(ret)))
        if pad_h or pad_w:
            out = out[:, :, :, :H, :W].contiguous()
        return out
