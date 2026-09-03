import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler
from core.submodules import sample_lowres_volume_to_quarter_5d

try:
    import corr_sampler
except:
    pass

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass

class CorrSampler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, coords, radius):
        ctx.save_for_backward(volume,coords)
        ctx.radius = radius
        corr, = corr_sampler.forward(volume, coords, radius)
        return corr
    @staticmethod
    def backward(ctx, grad_output):
        volume, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_volume, = corr_sampler.backward(volume, coords, grad_output, ctx.radius)
        return grad_volume, None, None


class CorrBlockFast1D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, **kwargs):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        # all pairs correlation
        corr = CorrBlockFast1D.corr(fmap1, fmap2)
        batch, h1, w1, dim, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, 1, w2)
        for i in range(self.num_levels):
            self.corr_pyramid.append(corr.view(batch, h1, w1, -1, w2//2**i))
            corr = F.avg_pool2d(corr, [1, 2], stride=[1, 2])

    def __call__(self, coords):
        out_pyramid = []
        bz, _, ht, wd = coords.shape
        coords = coords[:, [0]]
        for i in range(self.num_levels):
            corr = CorrSampler.apply(self.corr_pyramid[i].squeeze(3), coords/2**i, self.radius)
            out_pyramid.append(corr.view(bz, -1, ht, wd))
        return torch.cat(out_pyramid, dim=1)

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr / torch.sqrt(torch.tensor(D).float())


class PytorchAlternateCorrBlock1D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, **kwargs):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.fmap1 = fmap1
        self.fmap2 = fmap2

    def corr(self, fmap1, fmap2, coords):
        B, D, H, W = fmap2.shape
        # map grid coordinates to [-1,1]
        xgrid, ygrid = coords.split([1,1], dim=-1)
        xgrid = 2*xgrid/(W-1) - 1
        ygrid = 2*ygrid/(H-1) - 1

        grid = torch.cat([xgrid, ygrid], dim=-1)
        output_corr = []
        for grid_slice in grid.unbind(3):
            fmapw_mini = F.grid_sample(fmap2, grid_slice, align_corners=True)
            corr = torch.sum(fmapw_mini * fmap1, dim=1)
            output_corr.append(corr)
        corr = torch.stack(output_corr, dim=1).permute(0,2,3,1)

        return corr / torch.sqrt(torch.tensor(D).float())

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape
        fmap1 = self.fmap1
        fmap2 = self.fmap2
        out_pyramid = []
        for i in range(self.num_levels):
            dx = torch.zeros(1)
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)
            centroid_lvl = coords.reshape(batch, h1, w1, 1, 2).clone()
            centroid_lvl[..., 0] = centroid_lvl[..., 0] / 2**i
            coords_lvl = centroid_lvl + delta.view(-1, 2)
            corr = self.corr(fmap1, fmap2, coords_lvl)
            fmap2 = F.avg_pool2d(fmap2, [1, 2], stride=[1, 2])
            out_pyramid.append(corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()


class CorrBlock1D:
    def __init__(self, fmap1, fmap2, coords, num_levels=4, radius=4,
                 scale_list=[0.25, 0.5, 2.0, 4.0], scale_corr_radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.scale_list = scale_list
        self.scale_corr_radius = scale_corr_radius
        self.corr_pyramid = []
        self.coords_pyramid = []
        dx = torch.linspace(-radius, radius, 2*radius+1)
        self.dx = dx[:, None].to(coords.device)

        sdx = torch.linspace(-scale_corr_radius, scale_corr_radius, 2*scale_corr_radius+1)
        self.sdx = sdx[:, None].to(coords.device)

        # all pairs correlation
        corr = CorrBlock1D.corr(fmap1, fmap2)

        batch, h1, w1, _, w2 = corr.shape
        self.batch = batch
        self.h1 = h1
        self.w1 = w1
        self.w2 = w2
        corr = corr.reshape(batch*h1*w1, 1, 1, w2)
        self.coords = coords.reshape(batch*h1*w1, 1, 1, 1)

        self.corr_pyramid.append(corr)
        for i in range(1, self.num_levels):
            corr = F.avg_pool2d(corr, [1, 2], stride=[1, 2])
            self.corr_pyramid.append(corr)

    def __call__(self, disp, scaling=False):
        batch, _, h1, w1 = disp.shape
        disp = disp.reshape(self.batch*self.h1*self.w1, 1, 1, 1)
        out_pyramid = []

        if scaling:
            corr = self.corr_pyramid[0]
            for scale in self.scale_list:
                x0 = self.sdx + self.coords - scale * disp
                y0 = torch.zeros_like(x0)
                coords_lvl = torch.cat([x0, y0], dim=-1)
                corr_s = bilinear_sampler(corr, coords_lvl)
                corr_s = corr_s.view(self.batch, self.h1, self.w1, -1)
                out_pyramid.append(corr_s)
        else:
            coords = self.coords - disp
            for i in range(self.num_levels):
                corr = self.corr_pyramid[i]
                x0 = self.dx + coords / 2**i
                y0 = torch.zeros_like(x0)
                coords_lvl = torch.cat([x0, y0], dim=-1)
                corr_s = bilinear_sampler(corr, coords_lvl)
                corr_s = corr_s.view(self.batch, self.h1, self.w1, -1)
                out_pyramid.append(corr_s)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr / torch.sqrt(torch.tensor(D).float())


class CorrBlock1D1:
    def __init__(self, volume, coords, num_levels=4, radius=4,
                 scale_list=[0.25, 0.5, 2.0, 4.0], scale_corr_radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.scale_list = scale_list
        self.scale_corr_radius = scale_corr_radius
        self.volume_pyramid = []
        dx = torch.linspace(-radius, radius, 2*radius+1)
        self.dx = dx[:, None].to(coords.device)

        sdx = torch.linspace(-scale_corr_radius, scale_corr_radius, 2*scale_corr_radius+1)
        self.sdx = sdx[:, None].to(coords.device)

        # all pairs correlation
        batch, c, w2, h1, w1 = volume.shape
        self.batch = batch
        self.h1 = h1
        self.w1 = w1
        self.w2 = w2
        volume = volume.permute(0, 3, 4, 1, 2).reshape(batch*h1*w1, c, 1, w2)
        self.coords = coords.reshape(batch*h1*w1, 1, 1, 1)

        self.volume_pyramid.append(volume)
        for i in range(1, self.num_levels):
            volume = F.avg_pool2d(volume, [1, 2], stride=[1, 2])
            self.volume_pyramid.append(volume)

    def __call__(self, disp, scaling=False):
        #batch, _, h1, w1 = disp.shape
        disp = disp.float()  # Ensure disp is float32
        disp = disp.reshape(self.batch*self.h1*self.w1, 1, 1, 1)
        out_pyramid = []

        if scaling:
            volume = self.volume_pyramid[0]
            for scale in self.scale_list:
                x0 = self.sdx + scale * disp
                y0 = torch.zeros_like(x0)
                coords_lvl = torch.cat([x0, y0], dim=-1)
                volume_s = bilinear_sampler(volume, coords_lvl)
                volume_s = volume_s.view(self.batch, self.h1, self.w1, -1)
                out_pyramid.append(volume_s)
        else:
            for i in range(self.num_levels): 
                #corr = self.corr_pyramid[i]
                volume = self.volume_pyramid[i] 
                x0 = self.dx + disp / 2**i
                y0 = torch.zeros_like(x0)
                coords_lvl = torch.cat([x0, y0], dim=-1).float()
                #corr_s = bilinear_sampler(corr, coords_lvl)
                volume_s = bilinear_sampler(volume, coords_lvl)
                #corr_s = corr_s.view(self.batch, self.h1, self.w1, -1) #B, H, W, 1
                volume_s = volume_s.view(self.batch, self.h1, self.w1, -1) #B, H, W, 32
                #out_pyramid.append(corr_s)
                out_pyramid.append(volume_s)
        out_pyramid = torch.cat(out_pyramid, dim=-1) #B, H, W, 33 * (2*r+1) * num_levels/ 33 * (2*r+1) * len(scale_list)
        return out_pyramid.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = F.normalize(fmap1.view(B, D, H, W1), p=2, dim=1, eps=1e-6)
        fmap2 = F.normalize(fmap2.view(B, D, H, W2), p=2, dim=1, eps=1e-6)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr


class CorrBlock1D2:
    def __init__(self, volume, coords, feat1, feat2, num_levels=4, radius=4,
                 scale_list=[0.25, 0.5, 2.0, 4.0], scale_corr_radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.scale_list = scale_list
        self.scale_corr_radius = scale_corr_radius
        self.volume_pyramid = []
        dx = torch.linspace(-radius, radius, 2*radius+1)
        self.dx = dx[:, None].to(coords.device)
        self.corr_pyramid = []
        sdx = torch.linspace(-scale_corr_radius, scale_corr_radius, 2*scale_corr_radius+1)
        self.sdx = sdx[:, None].to(coords.device)

        corr = CorrBlock1D2.corr(feat1, feat2)
        # all pairs correlation
        batch, c, w2, h1, w1 = volume.shape
        _, _, _, w = feat1.shape
        self.batch = batch
        self.h1 = h1
        self.w1 = w1
        self.w2 = w2
        volume = volume.permute(0, 3, 4, 1, 2).reshape(batch*h1*w1, c, 1, w2)
        self.coords = coords.reshape(batch*h1*w1, 1, 1, 1)
        corr = corr.permute(0, 3, 4, 1, 2).reshape(batch*h1*w1, 1, 1, w)

        self.volume_pyramid.append(volume)
        self.corr_pyramid.append(corr)
        for i in range(1, self.num_levels):
            corr = F.avg_pool2d(corr, [1, 2], stride=[1, 2])
            self.corr_pyramid.append(corr)
            volume = F.avg_pool2d(volume, [1, 2], stride=[1, 2])
            self.volume_pyramid.append(volume)

    def __call__(self, disp, scaling=False):
        #batch, _, h1, w1 = disp.shape
        disp = disp.float()  # Ensure disp is float32
        disp = disp.reshape(self.batch*self.h1*self.w1, 1, 1, 1)
        out_pyramid = []

        if scaling:
            corr = self.corr_pyramid[0]
            volume = self.volume_pyramid[0]
            for scale in self.scale_list:
                x0 = self.sdx + scale * disp
                x1 = self.sdx + self.coords - scale * disp
                y0 = torch.zeros_like(x0)
                coords_lvl = torch.cat([x0, y0], dim=-1)
                coords_lvl2 = torch.cat([x1, y0], dim=-1)
                corr_s = bilinear_sampler(corr, coords_lvl2)
                corr_s = corr_s.view(self.batch, self.h1, self.w1, -1)
                volume_s = bilinear_sampler(volume, coords_lvl)
                volume_s = volume_s.view(self.batch, self.h1, self.w1, -1)
                out_pyramid.append(volume_s)
                out_pyramid.append(corr_s)
        else:
            coords = self.coords - disp
            for i in range(self.num_levels): 
                corr = self.corr_pyramid[i].float()
                volume = self.volume_pyramid[i] 
                x0 = self.dx + disp / 2**i
                x1 = self.dx + coords / 2**i
                y0 = torch.zeros_like(x0)
                coords_lvl = torch.cat([x0, y0], dim=-1)
                coords_lvl2 = torch.cat([x1, y0], dim=-1)
                corr_s = bilinear_sampler(corr, coords_lvl2)
                volume_s = bilinear_sampler(volume, coords_lvl)
                corr_s = corr_s.view(self.batch, self.h1, self.w1, -1) #B, H, W, 1
                volume_s = volume_s.view(self.batch, self.h1, self.w1, -1) #B, H, W, 32
                out_pyramid.append(volume_s)
                out_pyramid.append(corr_s)
        out_pyramid = torch.cat(out_pyramid, dim=-1) #B, H, W, 33 * (2*r+1) * num_levels/ 33 * (2*r+1) * len(scale_list)
        return out_pyramid.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr / torch.sqrt(torch.tensor(D).float())

class CorrBlock1D3:#加入了mask加权
    def __init__(self, volume, coords, mask, num_levels=4, radius=4,
                 scale_list=[0.25, 0.5, 2.0, 4.0], scale_corr_radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.scale_list = scale_list
        self.scale_corr_radius = scale_corr_radius
        self.volume_pyramid = []
        dx = torch.linspace(-radius, radius, 2*radius+1)
        self.dx = dx[:, None].to(coords.device)
        self.mask_pyramid = []
        sdx = torch.linspace(-scale_corr_radius, scale_corr_radius, 2*scale_corr_radius+1)
        self.sdx = sdx[:, None].to(coords.device)

        #self.mask = mask
        # all pairs correlation
        batch, c, w2, h1, w1 = volume.shape
        #_, _, _, _, w = feat1.shape
        self.batch = batch
        self.h1 = h1
        self.w1 = w1
        self.w2 = w2
        volume = volume.permute(0, 3, 4, 1, 2).reshape(batch*h1*w1, c, 1, w2)
        self.coords = coords.reshape(batch*h1*w1, 1, 1, 1)
        mask = mask.permute(0, 3, 4, 1, 2).reshape(batch*h1*w1, 1, 1, w2)

        self.volume_pyramid.append(volume)
        self.mask_pyramid.append(mask)
        for i in range(1, self.num_levels):
            mask = F.avg_pool2d(mask, [1, 2], stride=[1, 2])
            self.mask_pyramid.append(mask)
            volume = F.avg_pool2d(volume, [1, 2], stride=[1, 2])
            self.volume_pyramid.append(volume)

    def __call__(self, disp, scaling=False):
        #batch, _, h1, w1 = disp.shape
        disp = disp.float()  # Ensure disp is float32
        disp = disp.reshape(self.batch*self.h1*self.w1, 1, 1, 1)
        out_pyramid = []

        if scaling:
            corr = self.mask_pyramid[0]
            volume = self.volume_pyramid[0]
            for scale in self.scale_list:
                x0 = self.sdx + scale * disp
                #x1 = self.sdx + self.coords - scale * disp
                y0 = torch.zeros_like(x0)
                coords_lvl = torch.cat([x0, y0], dim=-1)
                #coords_lvl2 = torch.cat([x1, y0], dim=-1).half()
                corr_s = bilinear_sampler(corr, coords_lvl)
                #corr_s = corr_s.view(self.batch, self.h1, self.w1, -1)
                volume_s = bilinear_sampler(volume, coords_lvl) * (1 + corr_s)
                volume_s = volume_s.view(self.batch, self.h1, self.w1, -1)
                out_pyramid.append(volume_s)

        else:
            #coords = self.coords - disp
            for i in range(self.num_levels): 
                corr = self.mask_pyramid[i].float()
                volume = self.volume_pyramid[i] 
                x0 = self.dx + disp / 2**i
                #x1 = self.dx + coords / 2**i
                y0 = torch.zeros_like(x0)
                coords_lvl = torch.cat([x0, y0], dim=-1)
                #coords_lvl2 = torch.cat([x1, y0], dim=-1)
                corr_s = bilinear_sampler(corr, coords_lvl)
                volume_s = bilinear_sampler(volume, coords_lvl) * (1 + corr_s)
                #corr_s = corr_s.view(self.batch, self.h1, self.w1, -1) #B, H, W, 1
                volume_s = volume_s.view(self.batch, self.h1, self.w1, -1) #B, H, W, 32
                #out_pyramid.append(corr_s)
                out_pyramid.append(volume_s)
        out_pyramid = torch.cat(out_pyramid, dim=-1) #B, H, W, 33 * (2*r+1) * num_levels/ 33 * (2*r+1) * len(scale_list)
        return out_pyramid.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr / torch.sqrt(torch.tensor(D).float())

class CorrBlock1D4:#full costVolume Only
    def __init__(self, volume, coords, mask, num_levels=4, radius=4,
                 scale_list=[0.25, 0.5, 2.0, 4.0], scale_corr_radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.scale_list = scale_list
        self.scale_corr_radius = scale_corr_radius
        self.volume_pyramid = []
        dx = torch.linspace(-radius, radius, 2*radius+1)
        self.dx = dx[:, None].to(coords.device)
        self.mask_pyramid = []
        sdx = torch.linspace(-scale_corr_radius, scale_corr_radius, 2*scale_corr_radius+1)
        self.sdx = sdx[:, None].to(coords.device)

        #self.mask = mask
        # all pairs correlation
        batch, c, w2, h1, w1 = volume.shape
        #_, _, _, _, w = feat1.shape
        self.batch = batch
        self.h1 = h1
        self.w1 = w1
        self.w2 = w2
        volume = volume.permute(0, 3, 4, 1, 2).reshape(batch*h1*w1, c, 1, w2)
        self.coords = coords.reshape(batch*h1*w1, 1, 1, 1)
        mask = mask.permute(0, 3, 4, 1, 2).reshape(batch*h1*w1, 1, 1, w2)

        self.volume_pyramid.append(volume)
        self.mask_pyramid.append(mask)
        for i in range(1, self.num_levels):
            mask = F.avg_pool2d(mask, [1, 2], stride=[1, 2])
            self.mask_pyramid.append(mask)
            volume = F.avg_pool2d(volume, [1, 2], stride=[1, 2])
            self.volume_pyramid.append(volume)

    def __call__(self, disp, scaling=False):
        #batch, _, h1, w1 = disp.shape
        disp = disp.float()  # Ensure disp is float32
        disp = disp.reshape(self.batch*self.h1*self.w1, 1, 1, 1)
        out_pyramid = []

        if scaling:
            corr = self.mask_pyramid[0]
            volume = self.volume_pyramid[0]
            for scale in self.scale_list:
                x0 = self.sdx + scale * disp
                #x1 = self.sdx + self.coords - scale * disp
                y0 = torch.zeros_like(x0)
                coords_lvl = torch.cat([x0, y0], dim=-1)
                #coords_lvl2 = torch.cat([x1, y0], dim=-1).half()
                corr_s = bilinear_sampler(corr, coords_lvl)
                #corr_s = corr_s.view(self.batch, self.h1, self.w1, -1)
                volume_s = bilinear_sampler(volume, coords_lvl) * (1 + corr_s)
                volume_s = volume_s.view(self.batch, self.h1, self.w1, -1)
                out_pyramid.append(volume_s)

        else:
            #coords = self.coords - disp
            for i in range(self.num_levels): 
                corr = self.mask_pyramid[i].float()
                volume = self.volume_pyramid[i] 
                x0 = self.dx + disp / 2**i
                #x1 = self.dx + coords / 2**i
                y0 = torch.zeros_like(x0)
                coords_lvl = torch.cat([x0, y0], dim=-1)
                #coords_lvl2 = torch.cat([x1, y0], dim=-1)
                corr_s = bilinear_sampler(corr, coords_lvl)
                volume_s = bilinear_sampler(volume, coords_lvl) * (1 + corr_s)
                #corr_s = corr_s.view(self.batch, self.h1, self.w1, -1) #B, H, W, 1
                volume_s = volume_s.view(self.batch, self.h1, self.w1, -1) #B, H, W, 32
                #out_pyramid.append(corr_s)
                out_pyramid.append(volume_s)
        out_pyramid = torch.cat(out_pyramid, dim=-1) #B, H, W, 33 * (2*r+1) * num_levels/ 33 * (2*r+1) * len(scale_list)
        return out_pyramid.permute(0, 3, 1, 2).contiguous().float()

class CorrBlock1D5: #For fuse GWCVolume
    def __init__(self, volume, coords, feat1, feat2, num_levels=4, radius=4,
                 scale_list=[0.25, 0.5, 2.0, 4.0], scale_corr_radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.scale_list = scale_list
        self.scale_corr_radius = scale_corr_radius
        self.volume_pyramid = []
        dx = torch.linspace(-radius, radius, 2*radius+1)
        self.dx = dx[:, None].to(coords.device)
        self.corr_pyramid = []
        sdx = torch.linspace(-scale_corr_radius, scale_corr_radius, 2*scale_corr_radius+1)
        self.sdx = sdx[:, None].to(coords.device)

        corr = CorrBlock1D1.corr(feat1, feat2) #B, H, W1, 1, W2
        # all pairs correlation
        batch, c, w2, h1, w1 = volume.shape
        _, _, _, w = feat1.shape
        self.batch = batch
        self.h1 = h1
        self.w1 = w1
        self.w2 = w2
        volume = volume.permute(0, 3, 4, 1, 2).reshape(batch*h1*w1, c, 1, w2)
        self.coords = coords.reshape(batch*h1*w1, 1, 1, 1)
        corr = corr.reshape(batch*h1*w1, 1, 1, w)

        self.volume_pyramid.append(volume)
        self.corr_pyramid.append(corr)
        for i in range(1, self.num_levels):
            corr = F.avg_pool2d(corr, [1, 2], stride=[1, 2])
            self.corr_pyramid.append(corr)
            volume = F.avg_pool2d(volume, [1, 2], stride=[1, 2])
            self.volume_pyramid.append(volume)

    def __call__(self, disp, scaling=False):
        #batch, _, h1, w1 = disp.shape
        disp = disp.float()  # Ensure disp is float32
        disp = disp.reshape(self.batch*self.h1*self.w1, 1, 1, 1)
        out_pyramid = []

        if scaling:
            corr = self.corr_pyramid[0]
            volume = self.volume_pyramid[0]
            for scale in self.scale_list:
                x0 = self.sdx + scale * disp
                x1 = self.sdx + self.coords - scale * disp
                y0 = torch.zeros_like(x0)
                coords_lvl = torch.cat([x0, y0], dim=-1)
                coords_lvl2 = torch.cat([x1, y0], dim=-1)
                corr_s = bilinear_sampler(corr, coords_lvl2)
                corr_s = corr_s.view(self.batch, self.h1, self.w1, -1)
                volume_s = bilinear_sampler(volume, coords_lvl)
                volume_s = volume_s.view(self.batch, self.h1, self.w1, -1)
                out_pyramid.append(volume_s)
                out_pyramid.append(corr_s)
        else:
            coords = self.coords - disp
            for i in range(self.num_levels): 
                corr = self.corr_pyramid[i].float()
                volume = self.volume_pyramid[i] 
                x0 = self.dx / 4 + disp / 2**(i+2)    #disp to scale1
                x1 = self.dx + coords / 2**i
                y0 = torch.zeros_like(x0)
                coords_lvl = torch.cat([x0, y0], dim=-1)
                coords_lvl2 = torch.cat([x1, y0], dim=-1)
                corr_s = bilinear_sampler(corr, coords_lvl2)
                volume_s = bilinear_sampler(volume, coords_lvl)
                corr_s = corr_s.view(self.batch, self.h1, self.w1, -1) #B, H, W, 1
                volume_s = volume_s.view(self.batch, self.h1, self.w1, -1) #B, H, W, 32
                out_pyramid.append(volume_s)
                out_pyramid.append(corr_s)
        out_pyramid = torch.cat(out_pyramid, dim=-1) #B, H, W, 33 * (2*r+1) * num_levels/ 33 * (2*r+1) * len(scale_list)
        return out_pyramid.permute(0, 3, 1, 2).contiguous().float()

class CorrBlock1D6: # For 1/8 + 1/16 GWC volumes and 1/4 full correlation
    def __init__(self, gwcvolume1, gwcvolume2, corr, coords, num_levels=4, radius=4,
                 radius_in_disp_units=False):
        """
        Args:
            gwcvolume1: [B, C1, D1, H/8, W/8] GWC volume.
            gwcvolume2: [B, C2, D2, H/16, W/16] GWC volume.
            corr: [B, W_right, H/4, W_left/4] full correlation volume.
            coords: [B, 1, H/4, W_left/4] x-coordinate grid at update scale.
        """
        self.num_levels = num_levels
        self.radius = radius
        self.radius_in_disp_units = radius_in_disp_units
        self.gwcvolume1 = gwcvolume1
        self.gwcvolume2 = gwcvolume2
        self.corr_pyramid = []

        if gwcvolume1.ndim != 5:
            raise ValueError(f"gwcvolume1 must be [B,C,D,H,W], got {tuple(gwcvolume1.shape)}")
        if gwcvolume2.ndim != 5:
            raise ValueError(f"gwcvolume2 must be [B,C,D,H,W], got {tuple(gwcvolume2.shape)}")
        if corr.ndim != 4:
            raise ValueError(f"corr must be [B,W_right,H,W_left], got {tuple(corr.shape)}")
        if coords.ndim != 4 or coords.shape[1] != 1:
            raise ValueError(f"coords must be [B,1,H,W], got {tuple(coords.shape)}")

        batch, w2, h1, w1 = corr.shape
        if gwcvolume1.shape[0] != batch or gwcvolume2.shape[0] != batch or coords.shape[0] != batch:
            raise ValueError(
                "Batch size mismatch: "
                f"gwcvolume1={gwcvolume1.shape[0]}, "
                f"gwcvolume2={gwcvolume2.shape[0]}, corr={batch}, coords={coords.shape[0]}"
            )
        if coords.shape[-2:] != (h1, w1):
            raise ValueError(f"coords spatial size {tuple(coords.shape[-2:])} does not match corr {(h1, w1)}")

        self.batch = batch
        self.h1 = h1
        self.w1 = w1
        self.w2 = w2
        self.coords = coords.reshape(batch * h1 * w1, 1, 1, 1)

        dx = torch.linspace(-radius, radius, 2 * radius + 1)
        self.dx = dx[:, None].to(coords.device)

        corr = corr.permute(0, 2, 3, 1).contiguous().reshape(batch * h1 * w1, 1, 1, w2)
        self.corr_pyramid.append(corr)
        for _ in range(1, self.num_levels):
            corr = self._pool_corr_w(corr)
            self.corr_pyramid.append(corr)

    @staticmethod
    def _pool_corr_w(corr):
        if corr.shape[-1] <= 1:
            return corr
        return F.avg_pool2d(corr, [1, 2], stride=[1, 2])

    def __call__(self, disp):
        if disp.ndim != 4 or disp.shape[1] != 1:
            raise ValueError(f"disp must be [B,1,H,W], got {tuple(disp.shape)}")
        if disp.shape[0] != self.batch or disp.shape[-2:] != (self.h1, self.w1):
            raise ValueError(
                f"disp shape {tuple(disp.shape)} does not match "
                f"[{self.batch},1,{self.h1},{self.w1}]"
            )

        volume1_s = sample_lowres_volume_to_quarter_5d(
            self.gwcvolume1,
            disp,
            disp_divisor=2,
            radius=self.radius,
            radius_in_disp_units=self.radius_in_disp_units,
        )
        volume2_s = sample_lowres_volume_to_quarter_5d(
            self.gwcvolume2,
            disp,
            disp_divisor=4,
            radius=self.radius,
            radius_in_disp_units=self.radius_in_disp_units,
        )

        disp_flat = disp.float().reshape(self.batch * self.h1 * self.w1, 1, 1, 1)
        coords = self.coords - disp_flat
        corr_features = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i].float()
            x1 = self.dx + coords / 2 ** i
            y0 = torch.zeros_like(x1)
            coords_lvl = torch.cat([x1, y0], dim=-1)
            corr_s = bilinear_sampler(corr, coords_lvl)
            corr_s = corr_s.view(self.batch, self.h1, self.w1, -1)
            corr_s = corr_s.permute(0, 3, 1, 2).contiguous().float()
            corr_features.append(corr_s)

        return torch.cat([volume1_s, volume2_s] + corr_features, dim=1).contiguous().float()
        
    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = F.normalize(fmap1.view(B, D, H, W1), p=2, dim=1, eps=1e-6)
        fmap2 = F.normalize(fmap2.view(B, D, H, W2), p=2, dim=1, eps=1e-6)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr

class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, **kwargs):
        raise NotImplementedError
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(1, self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())
