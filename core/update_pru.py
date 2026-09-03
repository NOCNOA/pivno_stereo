import math
import os
import torch,pdb,sys
import torch.nn as nn
import torch.nn.functional as F
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from core.submodules import *
from core.extractor import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
from core.submodules import normalize_disparity


def _safe_normalize_disparity(disp, eps=1.0e-6):
    """PACT-safe robust normalization for constant disparity fields."""
    dtype = disp.dtype
    scale, shift = compute_scale_and_shift(disp.float())
    scale = torch.nan_to_num(
        scale, nan=1.0, posinf=1.0, neginf=1.0
    ).clamp_min(float(eps))
    shift = torch.nan_to_num(
        shift, nan=0.0, posinf=0.0, neginf=0.0
    )
    normalized = (
        disp.float() - shift[..., None, None]
    ) / scale[..., None, None]
    normalized = torch.nan_to_num(normalized)
    return normalized.to(dtype), scale.to(dtype), shift.to(dtype)

def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[-2:], **interp_args)

class DispHead(nn.Module):
    def __init__(self, hidden_dim):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.groups = 1
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)

        if self.bn == True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)

        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

class StructureEncoder(nn.Module):
    def __init__(self, hidden_dim, feat_dim):
        super(StructureEncoder, self).__init__()
        self.convc1 = nn.Conv2d(hidden_dim, hidden_dim // 2, 1, padding=0)
        self.convc2 = nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, padding=1)
        self.convd1 = nn.Conv2d(feat_dim, hidden_dim // 2, 7, padding=3)
        self.convd2 = nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, padding=1)
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)

    def forward(self, ctx, warp_feat):

        c = F.relu(self.convc1(ctx.float()), True)
        c = F.relu(self.convc2(c), True)

        d = F.relu(self.convd1(warp_feat), True)
        d = F.relu(self.convd2(d), True)
        out = F.relu(self.conv(torch.cat([c,d], dim=1)), True)

        return out

class MotionEncoder(nn.Module):
    def __init__(self, hidden_dim, corplane, warp_feat_dim):
        super(MotionEncoder, self).__init__()
        cor_plane = corplane
        self.convc1 = nn.Conv2d(cor_plane, hidden_dim // 2, 1, padding=0)
        self.convc2 = nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, padding=1)
        self.convd1 = nn.Conv2d(1, hidden_dim // 2, 7, padding=3)
        self.convd2 = nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, padding=1)
        self.convg1 = nn.Conv2d(warp_feat_dim, hidden_dim // 2, 1, padding=0)
        self.convg2 = nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, padding=1)
        self.conv = nn.Conv2d(hidden_dim*3//2, hidden_dim - 1, 3, padding=1)
    
    def forward(self, corr, disp, warp_feat):
        cor = F.relu(self.convc1(corr), True)
        cor = F.relu(self.convc2(cor), True)
        dis = F.relu(self.convd1(disp), True)
        dis = F.relu(self.convd2(dis), True)
        war = F.relu(self.convg1(warp_feat), True)
        war = F.relu(self.convg2(war), True)
        out = torch.cat([cor, dis, war], dim=1)
        out = F.relu(self.conv(out), True)

        return torch.cat([out, disp], dim=1)

class PromptStereoRecurrentUnit(nn.Module):
    def __init__(self, features, activation=nn.ReLU(False), deconv=False, bn=False, expand=False, align_corners=True, motion=False, size=None, use_res_input=True):
        super(PromptStereoRecurrentUnit, self).__init__()
        self.deconv = deconv
        self.expand = expand
        self.align_corners = align_corners
        self.size = size
        self.groups = 1

        out_features = features
        if self.expand == True:
            out_features = features // 2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.resConfUnit1 = ResidualConvUnit(features, activation, bn) if use_res_input else None
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        if motion:
            self.resConfUnitStructure = nn.Sequential(
                BasicConv(features, features, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0)
                )
            
            self.resConfUnitMotion = nn.Sequential(
                BasicConv(features, features, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0)
                )

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs, structure=None, motion=None, size=None):
        output = xs[0]

        if len(xs) == 2 and self.resConfUnit1 is not None:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if motion is not None:
            
            structure = self.resConfUnitStructure(structure)
            output = self.skip_add.add(output, structure)
            
            motion = self.resConfUnitMotion(motion)
            output = self.skip_add.add(output, motion)

        output = self.out_conv(output)

        return output


class MultiPromptUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim, feat_dim, volume_dim, warp_feat_dim, harddim=0,
                 adaptive_search=False, use_base_selection=True):
        """
        
        """
        super(MultiPromptUpdateBlock, self).__init__()
        self.adaptive_search = adaptive_search
        self.use_base_selection = bool(use_base_selection)
        self.stereo_pru = nn.ModuleList([
            PromptStereoRecurrentUnit(
                features=hidden_dim,
                bn=False,
                motion=(i == 0),
                use_res_input=(i != args.n_gru_layers - 1),
            )
            for i in range(args.n_gru_layers)
        ])
        self.structure_encoder = StructureEncoder(hidden_dim, 1)
        if harddim==0:
            cor_planes = args.corr_levels * (2*args.corr_radius + 1) * volume_dim
        else:
            cor_planes = harddim
        self.motion_encoder = MotionEncoder(hidden_dim, cor_planes, warp_feat_dim)
        self.disp_head = DispHead(hidden_dim)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, (2 ** 2 ** 2) * 9, 1, padding=0)
        )

        if self.adaptive_search:
            self.delta_info_head = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, 3, 3, padding=1),
            )
            nn.init.zeros_(self.delta_info_head[-1].weight)
            nn.init.zeros_(self.delta_info_head[-1].bias)
            with torch.no_grad():
                self.delta_info_head[-1].bias[2] = math.log(8.0)
            if self.use_base_selection:
                self.base_head = nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_dim, 3, 3, padding=1),
                )
                nn.init.zeros_(self.base_head[-1].weight)
                with torch.no_grad():
                    self.base_head[-1].bias.copy_(
                        torch.tensor([2.0, 0.0, 0.0])
                    )

        self.update = nn.ModuleList([
            nn.Sequential(
                BasicConv(hidden_dim * (2 + (i == 0)), hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()                
            ) for i in range(args.n_gru_layers)
        ])

    def forward(self, net, corr, disp, ctx, warp_feat, mono_disp):
        normalize = (
            _safe_normalize_disparity
            if self.adaptive_search
            else normalize_disparity
        )
        norm_mono, _, _ = normalize(mono_disp)
        norm_disp, _, _ = normalize(disp)
        norm_res = norm_mono - norm_disp
        structure = self.structure_encoder(ctx, norm_res)
        motion = self.motion_encoder(corr, disp, warp_feat)

        for i in reversed(range(len(net))):
            if (i == len(net) - 1):
                pooled = pool2x(net[i - 1])
                z = self.update[i](torch.cat([net[i], pooled], dim=1))
                pru_out = self.stereo_pru[i](net[i])
                net[i] = (1 - z) * net[i] + z * pru_out
            elif (i == 0):
                interp_feat = interp(net[i + 1], net[i])
                z = self.update[i](torch.cat([net[i], structure, motion], dim=1))
                pru_out = self.stereo_pru[i](net[i], interp_feat, structure=structure, motion=motion)
                net[i] = (1 - z) * net[i] + z * pru_out
            else:
                pooled = pool2x(net[i - 1])
                interp_feat = interp(net[i + 1], net[i])
                z = self.update[i](torch.cat([net[i], pooled], dim=1))
                pru_out = self.stereo_pru[i](net[i], interp_feat)
                net[i] = (1 - z) * net[i] + z * pru_out

        delta_disp = self.disp_head(net[0])
        mask = .25 * self.mask(net[0])

        if self.adaptive_search:
            delta_info = self.delta_info_head(net[0])
            if self.use_base_selection:
                base_logits = self.base_head(net[0])
                return net, delta_disp, mask, delta_info, base_logits
            return net, delta_disp, mask, delta_info

        return net, delta_disp, mask
