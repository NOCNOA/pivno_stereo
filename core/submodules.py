import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0, f"C:{C}, num_groups:{num_groups}"
    channels_per_group = C // num_groups
    fea1 = fea1.reshape(B, num_groups, channels_per_group, H, W)
    fea2 = fea2.reshape(B, num_groups, channels_per_group, H, W)
    with torch.cuda.amp.autocast(enabled=False):
      cost = (F.normalize(fea1.float(), dim=2) * F.normalize(fea2.float(), dim=2)).sum(dim=2)  #!NOTE Divide first for numerical stability
    assert cost.shape == (B, num_groups, H, W)
    return cost

def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.reshape(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)

def disparity_regression2(prob, maxdisp, interval=4):
    assert len(prob.shape) == 4
    disp_values = torch.arange(0, maxdisp, interval, dtype=prob.dtype, device=prob.device)
    disp_values = disp_values.view(1, maxdisp//interval, 1, 1)
    return torch.sum(prob * disp_values, 1, keepdim=True)

def context_upsample(disp_low, up_weights):
    """
    @disp_low: (b,1,h,w)  1/4 resolution
    @up_weights: (b,9,4*h,4*w)  Image resolution
    """
    b, c, h, w = disp_low.shape

    disp_unfold = F.unfold(disp_low.reshape(b,c,h,w),3,1,1).reshape(b,-1,h,w)
    disp_unfold = F.interpolate(disp_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)

    disp = (disp_unfold*up_weights).sum(1)

    return disp

class BasicConv_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, **kwargs):
        super(BasicConv_IN, self).__init__()

        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x

class hourglass(nn.Module):
    def __init__(self, cfg, in_channels, feat_dims=None):
        super().__init__()
        self.cfg = cfg
        self.corr_stem = BasicConv(in_channels, in_channels, is_3d=True, kernel_size=3, padding=1)
        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                    BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1))

        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17))

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, kernel_size=3, kernel_disp=17))


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        
        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, in_channels, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv_out = nn.Sequential(
          BasicConv(in_channels, in_channels, kernel_size=3, kernel_disp=17),
          BasicConv(in_channels, in_channels, kernel_size=3, kernel_disp=17),
        )

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17),
                                   BasicConv(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17),
                                   BasicConv(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17))

        self.conv_patch = nn.Sequential(
          nn.Conv3d(in_channels, in_channels, kernel_size=4, stride=4, padding=0, groups=in_channels),
          nn.BatchNorm3d(in_channels),
        )

        self.feature_att_8 = FeatureAtt(in_channels*2, feat_dims[1])
        self.feature_att_16 = FeatureAtt(in_channels*4, feat_dims[2])
        self.feature_att_32 = FeatureAtt(in_channels*6, feat_dims[3])
        self.feature_att_up_16 = FeatureAtt(in_channels*4, feat_dims[2])
        self.feature_att_up_8 = FeatureAtt(in_channels*2, feat_dims[1])

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        #print("ddffss", conv3_up.shape, conv2.shape)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)
        conv = self.conv_out(conv)

        return conv

class HourGlass(nn.Module):
    def __init__(self, cfg, volume_dim, feat_dim=None):
        super(HourGlass, self).__init__()
        cv_channel = volume_dim

        self.corr_stem = BasicConv(volume_dim, volume_dim, is_3d=True, kernel_size=3, padding=1)

        self.conv1 = nn.Sequential(
            BasicConv(cv_channel, cv_channel * 2, is_3d=True, kernel_size=3, stride=2, padding=1),
            BasicConv(cv_channel * 2, cv_channel * 2, is_3d=True, kernel_size=3, padding=1)
        )
        self.conv2 = nn.Sequential(
            BasicConv(cv_channel * 2, cv_channel * 4, is_3d=True, kernel_size=3, stride=2, padding=1),
            BasicConv(cv_channel * 4, cv_channel * 4, is_3d=True, kernel_size=3, padding=1)
        )
        self.conv3 = nn.Sequential(
            BasicConv(cv_channel * 4, cv_channel * 6, is_3d=True, kernel_size=3, stride=2, padding=1),
            BasicConv(cv_channel * 6, cv_channel * 6, is_3d=True, kernel_size=3, padding=1)
        )

        self.conv3_up = BasicConv(cv_channel * 6, cv_channel * 4, deconv=True, is_3d=True, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv2_up = BasicConv(cv_channel * 4, cv_channel * 2, deconv=True, is_3d=True, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv1_up = BasicConv(cv_channel * 2, cv_channel, deconv=True, is_3d=True, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))

        self.agg_0 = nn.Sequential(
            BasicConv(cv_channel * 8, cv_channel * 4, is_3d=True, kernel_size=1),
            BasicConv(cv_channel * 4, cv_channel * 4, is_3d=True, kernel_size=3, padding=1),
            BasicConv(cv_channel * 4, cv_channel * 4, is_3d=True, kernel_size=3, padding=1)
        )
        self.agg_1 = nn.Sequential(
            BasicConv(cv_channel * 4, cv_channel * 2, is_3d=True, kernel_size=1),
            BasicConv(cv_channel * 2, cv_channel * 2, is_3d=True, kernel_size=3, padding=1),
            BasicConv(cv_channel * 2, cv_channel * 2, is_3d=True, kernel_size=3, padding=1)
        )

        self.feature_att_4 = FeatureAtt(cv_channel, feat_dim[0])
        self.feature_att_8 = FeatureAtt(cv_channel * 2, feat_dim[1])
        self.feature_att_16 = FeatureAtt(cv_channel * 4, feat_dim[2])
        self.feature_att_32 = FeatureAtt(cv_channel * 6, feat_dim[3])
        self.feature_att_up_16 = FeatureAtt(cv_channel * 4, feat_dim[2])
        self.feature_att_up_8 = FeatureAtt(cv_channel * 2, feat_dim[1])

    def forward(self, x, feat):
        x = self.corr_stem(x)
        x = self.feature_att_4(x, feat[0])

        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, feat[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, feat[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, feat[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, feat[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, feat[1])

        conv = self.conv1_up(conv1)

        return conv

class ResnetBasicBlock(nn.Module):
  def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm2d, bias=False):
    super().__init__()
    self.norm_layer = norm_layer
    if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
    if self.norm_layer is not None:
      self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
    if self.norm_layer is not None:
      self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride
  def forward(self, x):
    identity = x
    out = self.conv1(x)
    if self.norm_layer is not None:
      out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    if self.norm_layer is not None:
      out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)
    out += identity
    out = self.relu(out)

    return out
class Conv2x_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, IN=True, relu=True, keep_dispc=False):
        super(Conv2x_IN, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = ResnetBasicBlock(out_channels*2, out_channels*mul, kernel_size=3, stride=1, padding=1, norm_layer=nn.InstanceNorm2d)
        else:
            self.conv2 = BasicConv_IN(out_channels, out_channels, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode='bilinear')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x

def groupwise_correlation2(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0, f"C:{C}, num_groups:{num_groups}"
    channels_per_group = C // num_groups
    fea1 = fea1.reshape(B, num_groups, channels_per_group, H, W)
    fea2 = fea2.reshape(B, num_groups, channels_per_group, H, W)
    cost = (fea1.float() * fea2.float()).mean(dim=2)  #!NOTE Divide first for numerical stability
    assert cost.shape == (B, num_groups, H, W)
    return cost

def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups = 8):
    """
    @refimg_fea: left image feature
    @targetimg_fea: right image feature
    """
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], num_groups)
        else:
            volume[:, :, i, :, :] =  groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume

def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):

    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, :] = refimg_fea[:, :, :, :]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume

def scale_coords(points, max_length):
    return torch.clamp(2 * points/(max_length-1.)- 1., -1., 1.)

def interpolate(feat, uv):
    uv = uv.transpose(1, 2) # feat: B, C, H, W
    uv = uv.unsqueeze(2)  # 1,Hx(W - startIdx), 2, 1
    samples = torch.nn.functional.grid_sample(feat, uv, mode='bilinear', padding_mode='border', align_corners=True)
    return samples[:, :, :, 0]


class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, norm='batch', **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        self.bn = nn.Identity()
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            if self.use_bn:
              if norm=='batch':
                self.bn = nn.BatchNorm3d(out_channels)
              elif norm=='instance':
                self.bn = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            if self.use_bn:
              if norm=='batch':
                self.bn = nn.BatchNorm2d(out_channels)
              elif norm=='instance':
                self.bn = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x

class ResnetBasicBlock3D(nn.Module):
  def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm3d, bias=False):
    super().__init__()
    self.norm_layer = norm_layer
    if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
    if self.norm_layer is not None:
      self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv3d(planes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
    if self.norm_layer is not None:
      self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride
  def forward(self, x):
    identity = x

    out = self.conv1(x)
    if self.norm_layer is not None:
      out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    if self.norm_layer is not None:
      out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)
    out += identity
    out = self.relu(out)
    return out

def disp_and_conf_from_prob_max(prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Use torch.max(dim=1) to compute disparity (argmax index) and confidence (max value).

    Args:
        prob: Tensor of shape [B, D, H, W], per-disparity probabilities (or scores).

    Returns:
        disp: Tensor [B, H, W], disparity index (argmax over D) as float32.
        conf: Tensor [B, H, W], max probability/value at the argmax disparity.
    """
    if prob.ndim != 4:
        raise ValueError(f"Expected prob with shape [B, D, H, W], got {tuple(prob.shape)}")

    conf, disp_idx = prob.max(dim=1)      # conf: [B,H,W], disp_idx: [B,H,W] (long)
    disp = disp_idx.to(torch.float32)     # [B,H,W]

    return disp.unsqueeze(1), conf.unsqueeze(1)


def window_prob_and_disp(prob: torch.Tensor, r: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:

    if prob.ndim != 4:
        raise ValueError(f"prob must be [B,D,H,W], got {tuple(prob.shape)}")
    if r < 0:
        raise ValueError("r must be >= 0")

    B, D, H, W = prob.shape
    K = 2 * r + 1

    # 1) 峰值：argmax 及其概率
    prob_peak, d_peak = prob.max(dim=1)                 # [B,H,W], [B,H,W] long
    prob_peak = prob_peak.unsqueeze(1)                  # [B,1,H,W]

    # 2) 峰值附近窗口 index：d_peak + offsets
    offsets = torch.arange(-r, r + 1, device=prob.device).view(1, K, 1, 1)  # [1,K,1,1]
    d_win = (d_peak.unsqueeze(1) + offsets).clamp(0, D - 1)                 # [B,K,H,W] long

    # 3) 取窗口概率
    prob_local = prob.gather(dim=1, index=d_win)        # [B,K,H,W]

    # 4) 局部 soft-argmax：窗口内期望（用窗口质量归一化，避免边界处和<1）
    mass = prob_local.sum(dim=1, keepdim=True).clamp_min(1e-8)              # [B,1,H,W]
    disp_local = (prob_local * d_win.float()).sum(dim=1, keepdim=True) / mass  # [B,1,H,W]

    return disp_local, mass


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=bn, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=bn, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode='bilinear')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x

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

        Q = Q.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        K = K.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        V = V.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)

        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, L, -1)
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
    self.conv_patch = nn.Sequential(
      nn.Conv3d(d_model, d_model, kernel_size=4, stride=4, padding=0),  #conv3d的group的作用是实现组卷积，可以减少参数量和计算量
      nn.BatchNorm3d(d_model),
    )

  def forward(self, cv, window_size=(-1,-1)):
    """
    @cv: (B,C,D,H,W) where D is max disparity
    """
    x = cv
    B, C, D_orig, H_orig, W_orig = x.shape
    x = self.conv_patch(x)
    B,C,D,H,W = x.shape

    x = x.permute(0,3,4,2,1).reshape(B*H*W, D, C)
    x = self.pos_embed0(x, resize_embed=self.resize_embed)  #!NOTE No resize since disparity is pre-determined
    for i in range(len(self.sa)):
        x = self.sa[i](x, window_size=window_size)
    x = x.reshape(B,H,W,D,C).permute(0,4,3,1,2)
    x = F.interpolate(x, size=(D_orig, H_orig, W_orig), mode='trilinear', align_corners=False)
    return x

class CostVolumeDisparityAttention2(nn.Module):
  def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, act=nn.GELU, norm_first=False, num_transformer=6, max_len=512, resize_embed=False):
    super().__init__()
    self.resize_embed = resize_embed
    self.sa = nn.ModuleList([])
    for _ in range(num_transformer):
      self.sa.append(FlashAttentionTransformerEncoderLayer(embed_dim=d_model, num_heads=nhead, dim_feedforward=dim_feedforward, act=act, dropout=dropout))
    self.pos_embed0 = PositionalEmbedding(d_model, max_len=max_len)
    self.conv_patch = nn.Sequential(
      nn.Conv3d(16, 16, kernel_size=4, stride=4, padding=0),  #conv3d的group的作用是实现组卷积，可以减少参数量和计算量
      nn.BatchNorm3d(16),
    )

  def forward(self, cv, window_size=(-1,-1)):
    """
    @cv: (B,C,D,H,W) where D is max disparity
    """
    x = cv
    x = self.conv_patch(x)
    B,C,D,H,W = x.shape

    x = x.permute(0,2,3,4,1).reshape(B*D, H*W, C) #B C D H W -> B D H W C
    x = self.pos_embed0(x, resize_embed=self.resize_embed)  #!NOTE No resize since disparity is pre-determined
    for i in range(len(self.sa)):
        x = self.sa[i](x, window_size=window_size)
    x = x.reshape(B,D,H,W,C).permute(0,4,1,2,3)
    x = F.interpolate(x, scale_factor=4, mode='trilinear', align_corners=False)
    return x

class LayerNorm2d(nn.LayerNorm):
    r""" https://huggingface.co/spaces/Roll20/pet_score/blob/b258ef28152ab0d5b377d9142a23346f863c1526/lib/timm/models/convnext.py#L85
    LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        """
        @x: (B,C,H,W)
        """
        if _is_contiguous(x):
            return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2).contiguous()
        else:
            s, u = torch.var_mean(x, dim=1, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x

class EdgeNextConvEncoder(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6, expan_ratio=4, kernel_size=7, norm='layer'):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        if norm=='layer':
          self.norm = LayerNorm2d(dim, eps=1e-6)
        else:
          self.norm = nn.Identity()
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x

class ChannelAttentionEnhancement(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttentionEnhancement, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttentionExtractor(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionExtractor, self).__init__()

        self.samconv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.samconv(x)
        return self.sigmoid(x)

class FeatureAtt(nn.Module):
    def __init__(self, cv_chan, feat_chan):
        super(FeatureAtt, self).__init__()

        self.feat_att = nn.Sequential(
            BasicConv(feat_chan, feat_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan//2, cv_chan, 1)
            )

    def forward(self, cv, feat):
        '''
        @cv: cost volume (B,C,D,H,W)
        @feat: (B,C,H,W)
        '''
        feat_att = self.feat_att(feat).unsqueeze(2)   #(B,cv_chan,1,H,W)
        #print("asdasdasd",feat_att.shape, cv.shape)
        cv = torch.sigmoid(feat_att)*cv
        return cv

class VolumeBinaryMaskHead(nn.Module):
    """
    Generate a binary (0/1) confidence mask from a stereo volume.

    Input : x      [B, C, D, H, W]   (concat+gwc merged volume)
    Output:
      - training: logits [B, 1, D, H, W]  (for BCEWithLogitsLoss etc.)
      - eval    : mask   [B, 1, D, H, W]  (uint8 0/1)

    Notes:
      - Binary output is only produced in eval() mode (model.training == False),
        because hard threshold is non-differentiable.
      - For thr=0.5, sigmoid(logits)>0.5 is equivalent to logits>0.
    """
    def __init__(
        self,
        in_channels: int,
        hidden: int = 64,
        use_3d_context: bool = True,
        norm: str = "gn",          # "gn" | "bn" | "none"
        gn_groups: int = 8,
        thr: float = 0.5,          # threshold for binarization in eval mode
        out_dtype: torch.dtype = torch.float32,  # torch.uint8 or torch.float32
    ):
        super().__init__()
        if not (0.0 < thr < 1.0):
            raise ValueError(f"thr must be in (0,1), got {thr}")
        self.thr = float(thr)
        self.out_dtype = out_dtype

        def make_norm(c: int):
            if norm == "gn":
                g = min(gn_groups, c)
                while c % g != 0 and g > 1:
                    g -= 1
                return nn.GroupNorm(g, c)
            elif norm == "bn":
                return nn.BatchNorm3d(c)
            elif norm == "none":
                return nn.Identity()
            else:
                raise ValueError(f"Unknown norm: {norm}")

        act = nn.SiLU(inplace=True)

        layers = []
        # voxel-wise channel mixing (MLP-like)
        layers += [
            nn.Conv3d(in_channels, hidden, kernel_size=1, bias=False),
            make_norm(hidden),
            act,
            nn.Conv3d(hidden, hidden, kernel_size=1, bias=False),
            make_norm(hidden),
            act,
        ]

        # optional light 3D context to smooth/regularize in D/H/W
        if use_3d_context:
            layers += [
                nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                make_norm(hidden),
                act,
                nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                make_norm(hidden),
                act,
            ]

        # final logits
        layers += [nn.Conv3d(hidden, 1, kernel_size=1, bias=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected x as [B,C,D,H,W], got {tuple(x.shape)}")

        logits = self.net(x)  # [B,1,D,H,W]

        # Eval: return binary 0/1 mask

        if self.thr == 0.5:
            mask = (logits > 0)
        else:
            mask = (torch.sigmoid(logits) > self.thr)

        if self.out_dtype == torch.uint8:
            return mask.to(torch.uint8)
        elif self.out_dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            return mask.to(self.out_dtype)
        else:
            raise ValueError(f"Unsupported out_dtype: {self.out_dtype}")

class Conv3dNormActReduced(nn.Module):
    def __init__(self, C_in, C_out, hidden=None, kernel_size=3, kernel_disp=None, stride=1, norm=nn.BatchNorm3d):
        super().__init__()
        if kernel_disp is None:
          kernel_disp = kernel_size
        if hidden is None:
            hidden = C_out
        self.conv1 = nn.Sequential(
            nn.Conv3d(C_in, hidden, kernel_size=(1,kernel_size,kernel_size), padding=(0, kernel_size//2, kernel_size//2), stride=(1, stride, stride)),
            norm(hidden),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(hidden, C_out, kernel_size=(kernel_disp, 1, 1), padding=(kernel_disp//2, 0, 0), stride=(stride, 1, 1)),
            norm(C_out),
            nn.ReLU(),
        )


    def forward(self, x):
        """
        @x: (B,C,D,H,W)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def compute_scale_and_shift(disp):
    flat_disp = disp.flatten(2)
    shift = torch.nanquantile(flat_disp, 0.5, dim=2)
    scale = torch.abs(flat_disp - shift[..., None]).nanmean(dim=2)
    shift[shift.isnan()] = 0
    scale[scale.isnan()] = 1

    return scale, shift

def normalize_disparity(disp):
    dtype = disp.dtype
    scale, shift = compute_scale_and_shift(disp.float())
    norm_disp = (disp - shift[..., None, None]) / scale[..., None, None]

    return norm_disp.to(dtype), scale.to(dtype), shift.to(dtype)


def sample_lowres_volume_to_quarter_5d(
    volume,
    disp,
    disp_divisor,
    radius,
    radius_in_disp_units=False,
    padding_mode="zeros",
    align_corners=True,
):
    """
    Sample a low-resolution cost volume directly at the spatial resolution of
    ``disp`` using 5D grid_sample.

    Args:
        volume: [B, C, D, H_low, W_low] cost/geometry volume.
        disp: [B, 1, H_out, W_out] disparity in the 1/4-resolution update units.
        disp_divisor: maps disp units to the volume D-axis units. Use 2 for a
            1/8 volume and 4 for a 1/16 volume when disp is in 1/4 units.
        radius: local D-axis sampling radius.
        radius_in_disp_units: if False, offsets are volume D bins. If True,
            offsets are in disp units and are divided by disp_divisor.

    Returns:
        [B, C * (2 * radius + 1), H_out, W_out]
    """
    if volume.ndim != 5:
        raise ValueError(f"Expected volume as [B,C,D,H,W], got {tuple(volume.shape)}")
    if disp.ndim != 4 or disp.shape[1] != 1:
        raise ValueError(f"Expected disp as [B,1,H,W], got {tuple(disp.shape)}")
    if volume.shape[0] != disp.shape[0]:
        raise ValueError(f"Batch size mismatch: volume {volume.shape[0]} vs disp {disp.shape[0]}")
    if disp_divisor <= 0:
        raise ValueError(f"disp_divisor must be positive, got {disp_divisor}")

    b, c, d, h_low, w_low = volume.shape
    _, _, h_out, w_out = disp.shape
    dtype = disp.dtype if disp.is_floating_point() else volume.dtype
    device = disp.device

    yy, xx = torch.meshgrid(
        torch.arange(h_out, device=device, dtype=torch.float32),
        torch.arange(w_out, device=device, dtype=torch.float32),
        indexing="ij",
    )

    if align_corners:
        if w_out > 1:
            x_low = xx * (w_low - 1) / (w_out - 1)
        else:
            x_low = xx.new_full((h_out, w_out), 0.5 * (w_low - 1))
        if h_out > 1:
            y_low = yy * (h_low - 1) / (h_out - 1)
        else:
            y_low = yy.new_full((h_out, w_out), 0.5 * (h_low - 1))
    else:
        x_low = (xx + 0.5) * w_low / w_out - 0.5
        y_low = (yy + 0.5) * h_low / h_out - 0.5

    k = 2 * radius + 1
    offsets = torch.linspace(-radius, radius, k, device=device, dtype=torch.float32).view(1, k, 1, 1)
    offset_scale = 1.0 / float(disp_divisor) if radius_in_disp_units else 1.0

    z_low = disp.float()[:, 0:1] / float(disp_divisor) + offsets * offset_scale

    def normalize_grid(coord, size):
        if size > 1:
            return 2.0 * coord / (size - 1) - 1.0
        return torch.zeros_like(coord)

    x_norm = normalize_grid(x_low, w_low).view(1, 1, h_out, w_out).expand(b, k, h_out, w_out)
    y_norm = normalize_grid(y_low, h_low).view(1, 1, h_out, w_out).expand(b, k, h_out, w_out)
    z_norm = normalize_grid(z_low, d)

    # 5D grid_sample expects grid coordinates in x, y, z order, mapping to
    # input dimensions W, H, D respectively for an input [B,C,D,H,W].
    grid = torch.stack([x_norm, y_norm, z_norm], dim=-1)

    sampled = F.grid_sample(
        volume.float(),
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    return sampled.reshape(b, c * k, h_out, w_out).to(dtype)

def _squeeze_full_corr(corr_full):
    """Normalize row-wise full correlation to [B, H, W_left, W_right]."""
    if corr_full.dim() == 5:
        if corr_full.shape[3] != 1:
            raise ValueError(f"Expected corr_full [B,H,W,1,W], got {tuple(corr_full.shape)}")
        corr_full = corr_full.squeeze(3)
    if corr_full.dim() != 4:
        raise ValueError(f"corr_full must be [B,H,W_left,W_right], got {tuple(corr_full.shape)}")
    return corr_full

def full_corr_to_disp_volume(corr_full, invalid_value=-1e4):
    """Convert right-coordinate full correlation into disparity-index volume.

    Args:
        corr_full: [B, H, W_left, W_right] or [B, H, W_left, 1, W_right].
        invalid_value: fill value for invalid disparity samples.

    Returns:
        disp_volume: [B, D, H, W_left], where D == W_right and
            disp_volume[:, d, :, x] stores corr(left_x=x, right_x=x-d).
        valid: [B, D, H, W_left] boolean mask for valid disparity samples.
    """
    corr_full = _squeeze_full_corr(corr_full)
    B, H, W_left, W_right = corr_full.shape
    device = corr_full.device

    disp_values = torch.arange(W_right, device=device).view(1, W_right, 1, 1)
    left_x = torch.arange(W_left, device=device).view(1, 1, 1, W_left)
    right_x = left_x - disp_values
    valid = (right_x >= 0) & (right_x < W_right)

    gather_index = right_x.clamp(0, W_right - 1).expand(B, -1, H, -1).long()
    volume = corr_full.permute(0, 3, 1, 2).contiguous()
    disp_volume = torch.gather(volume, dim=1, index=gather_index)
    valid = valid.expand_as(disp_volume)
    disp_volume = disp_volume.masked_fill(~valid, invalid_value)
    return disp_volume, valid


def local_softargmax_from_volume(disp_volume, valid=None, radius=4, temperature=0.1, invalid_value=-1e4):
    """Local soft-argmax on a disparity-index correlation volume."""
    if disp_volume.dim() != 4:
        raise ValueError(f"disp_volume must be [B,D,H,W], got {tuple(disp_volume.shape)}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    B, D, H, W = disp_volume.shape
    device = disp_volume.device
    dtype = disp_volume.dtype
    volume = disp_volume.float()
    if valid is None:
        valid = torch.ones_like(disp_volume, dtype=torch.bool)
    else:
        valid = valid.to(device=device, dtype=torch.bool)
    volume = volume.masked_fill(~valid, invalid_value)

    peak_disp = volume.argmax(dim=1)
    offsets = torch.arange(-radius, radius + 1, device=device).view(1, -1, 1, 1)
    local_disp = peak_disp.unsqueeze(1) + offsets
    local_valid = (local_disp >= 0) & (local_disp < D)

    gather_index = local_disp.clamp(0, D - 1).long()
    local_logits = torch.gather(volume, dim=1, index=gather_index)
    local_valid = local_valid & torch.gather(valid, dim=1, index=gather_index)
    local_logits = local_logits.masked_fill(~local_valid, invalid_value)

    prob = torch.softmax(local_logits / temperature, dim=1)
    prob = prob * local_valid.float()
    prob = prob / prob.sum(dim=1, keepdim=True).clamp_min(1e-8)

    d0 = torch.sum(prob * local_disp.float(), dim=1, keepdim=True).to(dtype)
    topk = torch.topk(volume, k=min(2, D), dim=1)
    top1_score = topk.values[:, :1].to(dtype)
    top1_margin = (topk.values[:, :1] - topk.values[:, 1:2]).to(dtype) if D > 1 else torch.zeros_like(top1_score)
    entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=1, keepdim=True).to(dtype)
    confidence = {
        "top1_score": top1_score,
        "top1_margin": top1_margin,
        "entropy": entropy,
        "peak_disp": peak_disp.unsqueeze(1).to(dtype),
    }
    return d0, confidence

class RightWidthCompressor(nn.Module):
    """
    Compress right feature along width dimension only.

    Input:
        right_feat: [B, C, H, W]

    Output:
        right_feat_c: [B, C_out, H, Wc]

    Notes:
        - Only the width dimension is compressed.
        - With ceil_mode=False, Wc = W // compress_ratio.
        - mode="conv" uses learnable width-stride convolution:
          kernel_size=(1, compress_ratio), stride=(1, compress_ratio).
        - Coordinates in the compressed right feature correspond roughly to
          x_right ~= j * compress_ratio in the input feature scale.
    """

    def __init__(
        self,
        compress_ratio=4,
        mode="conv",
        in_channels=None,
        out_channels=None,
        refine=False,
    ):
        super().__init__()
        if compress_ratio < 1:
            raise ValueError(f"compress_ratio must be >= 1, got {compress_ratio}")
        if mode not in ("conv", "depthwise_conv", "avg", "max"):
            raise ValueError(f"mode must be 'conv', 'depthwise_conv', 'avg' or 'max', got {mode}")
        if mode in ("conv", "depthwise_conv") and in_channels is None:
            raise ValueError("in_channels is required when using learnable compression")
        if refine and in_channels is None:
            raise ValueError("in_channels is required when refine=True")
        if mode in ("avg", "max") and out_channels is not None and in_channels is not None and out_channels != in_channels:
            raise ValueError("out_channels can only differ from in_channels for learnable compression modes")

        self.compress_ratio = int(compress_ratio)
        self.mode = mode

        compressed_channels = out_channels if out_channels is not None else in_channels

        if mode == "conv":
            self.compress = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    compressed_channels,
                    kernel_size=(1, self.compress_ratio),
                    stride=(1, self.compress_ratio),
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(compressed_channels),
                nn.ReLU(inplace=True),
            )
        elif mode == "depthwise_conv":
            self.compress = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=(1, self.compress_ratio),
                    stride=(1, self.compress_ratio),
                    padding=0,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, compressed_channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(compressed_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.compress = None
            compressed_channels = in_channels

        if refine:
            self.refine = nn.Sequential(
                nn.Conv2d(compressed_channels, compressed_channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(compressed_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.refine = nn.Identity()

    def forward(self, right_feat):
        if right_feat.ndim != 4:
            raise ValueError(f"Expected right_feat as [B,C,H,W], got {tuple(right_feat.shape)}")

        _, _, _, width = right_feat.shape
        if width < self.compress_ratio:
            raise ValueError(
                f"Input width {width} is smaller than compress_ratio {self.compress_ratio}"
            )

        if self.mode in ("conv", "depthwise_conv"):
            right_feat_c = self.compress(right_feat)
        elif self.compress_ratio == 1:
            right_feat_c = right_feat
        elif self.mode == "avg":
            right_feat_c = F.avg_pool2d(
                right_feat,
                kernel_size=(1, self.compress_ratio),
                stride=(1, self.compress_ratio),
                ceil_mode=False,
            )
        else:
            right_feat_c = F.max_pool2d(
                right_feat,
                kernel_size=(1, self.compress_ratio),
                stride=(1, self.compress_ratio),
                ceil_mode=False,
            )

        return self.refine(right_feat_c)

class ParallelRightWidthCompressor(nn.Module):
    """Build half-width and quarter-width right-feature branches.

    Both branches operate directly on the same input instead of being
    cascaded:

        right_feat:   [B, C_in,  H, W]
        right_feat_2: [B, C_out, H, W // 2]
        right_feat_4: [B, C_out, H, W // 4]

    With zero padding and ``kernel_size == stride == ratio``, compressed
    coordinate ``j`` represents a receptive-field center at
    ``ratio * j + (ratio - 1) / 2`` in the input feature coordinate system.
    This mapping should be used when the dynamic sampler is connected later.
    """

    def __init__(self, in_channels, out_channels=None, mode="conv"):
        super().__init__()
        if not isinstance(in_channels, int) or in_channels < 1:
            raise ValueError(f"in_channels must be a positive integer, got {in_channels}")
        if out_channels is None:
            out_channels = in_channels
        if not isinstance(out_channels, int) or out_channels < 1:
            raise ValueError(f"out_channels must be a positive integer, got {out_channels}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode = mode
        self.compress_2 = RightWidthCompressor(
            compress_ratio=2,
            mode=mode,
            in_channels=in_channels,
            out_channels=out_channels,
        )
        self.compress_4 = RightWidthCompressor(
            compress_ratio=4,
            mode=mode,
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def forward(self, right_feat):
        if right_feat.ndim != 4:
            raise ValueError(f"Expected right_feat as [B,C,H,W], got {tuple(right_feat.shape)}")
        if right_feat.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {right_feat.shape[1]}"
            )
        if right_feat.shape[-1] < 4:
            raise ValueError(f"Input width must be at least 4, got {right_feat.shape[-1]}")

        right_feat_2 = self.compress_2(right_feat)
        right_feat_4 = self.compress_4(right_feat)
        return right_feat_2, right_feat_4

def sample_right_feature_pyramid(right_features, disp, offsets, compression_ratios=(1, 2, 4), padding_mode='zeros', align_corners=True):
    """Sample right features around the current left-view disparity.

    Args:
        right_features: sequence of [B,C,H,W_r] tensors. Each tensor must be
            compressed directly from the original right feature by the
            corresponding ratio.
        disp: current disparity [B,1,H,W] in the uncompressed 1/4-feature unit.
        offsets: one-dimensional offsets in each branch's compressed-cell unit.
        compression_ratios: width compression ratio for every right feature.

    Returns:
        Sampled right features [B, num_scales*num_offsets, C, H, W].

    For a width-stride convolution with kernel_size == stride == ratio,
    compressed coordinate j has its center at
    ratio*j + (ratio-1)/2 in the original feature coordinates. Therefore the
    right lookup coordinate is:

        j = (x_left - disp - (ratio-1)/2) / ratio + offset
    """
    if disp.ndim != 4 or disp.shape[1] != 1:
        raise ValueError(f"disp must be [B,1,H,W], got {tuple(disp.shape)}")
    if len(right_features) != len(compression_ratios):
        raise ValueError(f"Expected one compression ratio per feature, got {len(right_features)} features and {len(compression_ratios)} ratios")

    batch, _, query_height, query_width = disp.shape
    offsets = torch.as_tensor(offsets, device=disp.device, dtype=disp.dtype).flatten()
    if offsets.numel() == 0:
        raise ValueError("offsets must contain at least one value")

    x_left = torch.arange(query_width, device=disp.device, dtype=disp.dtype).view(1, 1, 1, query_width)
    y = torch.arange(query_height, device=disp.device, dtype=disp.dtype).view(1, 1, query_height, 1)
    sampled_scales = []

    for right_feat, ratio in zip(right_features, compression_ratios):
        if right_feat.ndim != 4:
            raise ValueError(f"Every right feature must be [B,C,H,W], got {tuple(right_feat.shape)}")
        if right_feat.shape[0] != batch or right_feat.shape[2] != query_height:
            raise ValueError(f"Right feature batch/height must match disp, got {tuple(right_feat.shape)} and {tuple(disp.shape)}")
        if ratio < 1:
            raise ValueError(f"compression ratios must be positive, got {ratio}")

        source_width = right_feat.shape[-1]
        source_height = right_feat.shape[-2]
        center_offset = (float(ratio) - 1.0) / 2.0
        x_right = (x_left - disp - center_offset) / float(ratio)
        x_right = x_right + offsets.view(1, -1, 1, 1)
        y_right = y.expand(batch, offsets.numel(), query_height, query_width)

        if align_corners:
            x_norm = 2.0 * x_right / max(source_width - 1, 1) - 1.0
            y_norm = 2.0 * y_right / max(source_height - 1, 1) - 1.0
        else:
            x_norm = 2.0 * (x_right + 0.5) / source_width - 1.0
            y_norm = 2.0 * (y_right + 0.5) / source_height - 1.0

        sample_count = offsets.numel()
        grid = torch.stack([x_norm, y_norm], dim=-1).reshape(batch * sample_count, query_height, query_width, 2)
        channels = right_feat.shape[1]
        expanded_right = right_feat.unsqueeze(1).expand(batch, sample_count, channels, source_height, source_width)
        expanded_right = expanded_right.reshape(batch * sample_count, channels, source_height, source_width)

        with torch.cuda.amp.autocast(enabled=False):
            sampled = F.grid_sample(
                expanded_right.float(),
                grid.float(),
                mode='bilinear',
                padding_mode=padding_mode,
                align_corners=align_corners,
            )
        sampled_scales.append(sampled.to(right_feat.dtype).reshape(batch, sample_count, channels, query_height, query_width))

    return torch.cat(sampled_scales, dim=1) #batch, sample_count*level, channels, query_height, query_width

def lookup_right_features(fmap1_low, right_feature_pyramid, disp, warp_offsets, feature_fuse, rope_galerkin=None, scale_weights=None, compression_ratios=(1, 2, 4)):
    """Sample and fuse local right-feature windows at the current disparity.

    Grid sampling stays in FP32 inside sample_right_feature_pyramid. The
    sampled tensors are converted back to the feature dtype before fusion so
    the large window and attention activations can use mixed precision.

    Without scale weights, all ``S`` scale branches are concatenated and
    ``feature_fuse`` receives ``C + S*K*C`` channels. With scale weights, the
    original weighted reduction is retained and it receives ``C + K*C``.
    """
    if fmap1_low.ndim != 4:
        raise ValueError(f"fmap1_low must be [B,C,H,W], got {tuple(fmap1_low.shape)}")
    if len(right_feature_pyramid) != len(compression_ratios):
        raise ValueError("right_feature_pyramid and compression_ratios must have equal length")

    sampled_right = sample_right_feature_pyramid(right_feature_pyramid, disp, offsets=warp_offsets.flatten(), compression_ratios=compression_ratios, padding_mode='zeros', align_corners=True)
    batch, channels, height, width = fmap1_low.shape
    num_scales = len(compression_ratios)
    num_samples = sampled_right.shape[1]
    if num_samples % num_scales != 0:
        raise RuntimeError(f"Cannot split {num_samples} samples into {num_scales} scales")

    samples_per_scale = num_samples // num_scales
    sampled_right = sampled_right.reshape(batch, num_scales, samples_per_scale, channels, height, width)

    if scale_weights is None:
        right_for_fusion = sampled_right.flatten(1, 3)
    else:
        expected = (batch, num_scales, height, width)
        if tuple(scale_weights.shape) != expected:
            raise ValueError(f"scale_weights must be {expected}, got {tuple(scale_weights.shape)}")
        selected_right = (sampled_right * scale_weights.to(sampled_right.dtype)[:, :, None, None]).sum(dim=1)
        right_for_fusion = selected_right.flatten(1, 2)
    fused_input = torch.cat([fmap1_low.to(sampled_right.dtype), right_for_fusion], dim=1)
    del sampled_right, right_for_fusion
    fused_feature = feature_fuse(fused_input)
    del fused_input
    if rope_galerkin is not None:
        fused_feature = rope_galerkin(fused_feature)

    return fused_feature
