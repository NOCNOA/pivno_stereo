import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import timm
from depth_anything_v2.dpt import DepthAnythingV2
from core.submodules import BasicConv, Conv2x_IN
class Feature(nn.Module):
    def __init__(self, args, out_dim):
        super(Feature, self).__init__()
        self.args = args
        model = timm.create_model('edgenext_small', pretrained=False, checkpoint_path='checkpoints/pytorch_model.bin', features_only=False)
        self.stem = model.stem
        self.stages = model.stages
        chans = [48, 96, 160, 304]
        self.chans = chans
        self.deconv32_16 = Conv2x_IN(chans[3], chans[2], deconv=True, concat=True)
        self.deconv16_8 = Conv2x_IN(chans[2]*2, chans[1], deconv=True, concat=True)
        self.deconv8_4 = Conv2x_IN(chans[1]*2, chans[0], deconv=True, concat=True)
        vit_feat_dim = 64
        self.conv4 = nn.Sequential(
          BasicConv(chans[0]*2+vit_feat_dim, chans[0]*2+vit_feat_dim, kernel_size=3, stride=1, padding=1, norm='instance'),
          ResidualBlock(chans[0]*2+vit_feat_dim, chans[0]*2+vit_feat_dim, norm_fn='instance'),
          ResidualBlock(chans[0]*2+vit_feat_dim, out_dim, norm_fn='instance'),
        )

        #self.patch_size = 14
        self.d_out = [chans[0]*2+vit_feat_dim, chans[1]*2, chans[2]*2, chans[3]] #96+128 = 224

    def forward(self, x, vit_feat):
        B,C,H,W = x.shape
        B = B//2
        x = self.stem(x)
        x4 = self.stages[0](x)
        x8 = self.stages[1](x4)
        x16 = self.stages[2](x8)
        x32 = self.stages[3](x16)

        x16 = self.deconv32_16(x32, x16)
        x8 = self.deconv16_8(x16, x8)
        x4 = self.deconv8_4(x8, x4)
        #print("asd", x4.shape, vit_feat.shape)
        x4 = torch.cat([x4, vit_feat], dim=1)
        x4 = self.conv4(x4)
        return x4[:B], x4[B:]

class ConvBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

    def forward(self, x):

        return self.relu(self.norm1(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1, ratio=4):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes // ratio, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes // ratio, planes // ratio, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes // ratio, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // ratio)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // ratio)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes // ratio)
            self.norm2 = nn.BatchNorm2d(planes // ratio)
            self.norm3 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm4 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes // ratio)
            self.norm2 = nn.InstanceNorm2d(planes // ratio)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm4 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, d_dim, output_dim=128, norm_fn='batch', downsample=3):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))

        # depth feat convolution
        self.convd = ConvBlock(d_dim, 128, self.norm_fn)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, dfeats):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        is_list = isinstance(dfeats, tuple) or isinstance(dfeats, list)
        if is_list:
            batch_dim = dfeats[0].shape[0]
            dfeats = torch.cat(dfeats, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x + self.convd(dfeats)

        x = self.conv2(x)

        if is_list:
            x = x.split(split_size=batch_dim, dim=0)

        return x

class BasicEncoder2(nn.Module):
    def __init__(self, d_dim, output_dim=128, norm_fn='batch', downsample=2):
        super(BasicEncoder2, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if isinstance(output_dim, int):
            self.d_out = [output_dim] * 4
        else:
            if len(output_dim) != 4:
                raise ValueError(f"BasicEncoder2 expects 4 output dims, got {output_dim}")
            self.d_out = list(output_dim)

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)
        self.layer6 = self._make_layer(128, stride=2)

        # Depth feature fusion happens at the 1/4 stage, then the deeper pyramid
        # levels are derived from the fused representation.
        self.convd = ConvBlock(d_dim, 128, self.norm_fn)
        self.out08 = nn.Conv2d(128, self.d_out[0], kernel_size=1)
        self.out16 = nn.Conv2d(128, self.d_out[1], kernel_size=1)
        self.out32 = nn.Conv2d(128, self.d_out[2], kernel_size=1)
        self.out64 = nn.Conv2d(128, self.d_out[3], kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, dfeats, num_layers=4):
        if num_layers < 1 or num_layers > 4:
            raise ValueError(f"num_layers must be in [1, 4], got {num_layers}")

        split_lr = isinstance(x, (tuple, list))
        if split_lr:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        if isinstance(dfeats, (tuple, list)):
            dfeats = torch.cat(dfeats, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        feat_4 = x + self.convd(dfeats)
        pyramids = [self.out08(feat_4)]
        if num_layers == 1:
            if split_lr:
                return [pyramids[0][:batch_dim]], [pyramids[0][batch_dim:]]
            return pyramids

        feat_8 = self.layer4(feat_4)
        pyramids.append(self.out16(feat_8))
        if num_layers == 2:
            if split_lr:
                return ([feat[:batch_dim] for feat in pyramids],
                        [feat[batch_dim:] for feat in pyramids])
            return pyramids

        feat_16 = self.layer5(feat_8)
        pyramids.append(self.out32(feat_16))
        if num_layers == 3:
            if split_lr:
                return ([feat[:batch_dim] for feat in pyramids],
                        [feat[batch_dim:] for feat in pyramids])
            return pyramids

        feat_32 = self.layer6(feat_16)
        pyramids.append(self.out64(feat_32))

        if split_lr:
            left = [feat[:batch_dim] for feat in pyramids]
            right = [feat[batch_dim:] for feat in pyramids]
            return left, right

        return pyramids

class MultiBasicEncoder(nn.Module):
    def __init__(self, d_dim, output_dim=[128, 128, 128], norm_fn='batch', downsample=3, drop_path_rate=0.1):
        super(MultiBasicEncoder, self).__init__()
        self.d_dim = d_dim
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)

        self.drop_path = DropPath(drop_path_rate)

        self.conv08 = ConvBlock(d_dim, 128, self.norm_fn)
        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        self.conv16 = ConvBlock(d_dim, 128, self.norm_fn)
        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

        self.conv32 = ConvBlock(d_dim, 128, self.norm_fn)
        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs32 = nn.ModuleList(output_list)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, d_feats, num_layers=3, output_counts=None):

        if output_counts is None:
            output_counts = (len(self.outputs08), len(self.outputs16), len(self.outputs32))
        if len(output_counts) != 3:
            raise ValueError("output_counts must contain one count per pyramid level")

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        feat = x + self.drop_path(self.conv08(d_feats[0]))
        outputs08 = [f(feat) for f in self.outputs08[:output_counts[0]]]
        if num_layers == 1:
            return (outputs08,)

        y = self.layer4(x)
        feat = y + self.drop_path(self.conv16(d_feats[1]))
        outputs16 = [f(feat) for f in self.outputs16[:output_counts[1]]]

        if num_layers == 2:
            return (outputs08, outputs16)

        z = self.layer5(y)
        feat = z + self.drop_path(self.conv32(d_feats[2]))
        outputs32 = [f(feat) for f in self.outputs32[:output_counts[2]]]

        return (outputs08, outputs16, outputs32)


class DefomEncoder(nn.Module):
    def __init__(self, dinov2_encoder, pretrained=True, freeze=True, idepth_scale=0.25):
        super(DefomEncoder, self).__init__()
        self.dinov2_encoder = dinov2_encoder
        self.idepth_scale = idepth_scale
        self.pretrained = pretrained
        self.freeze = freeze

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        self.depth_anything = DepthAnythingV2(**model_configs[self.dinov2_encoder])

        if pretrained and os.path.exists(f'./checkpoints/depth_anything_v2_{dinov2_encoder}.pth'):
            self.depth_anything.load_state_dict(
                torch.load(f'./checkpoints/depth_anything_v2_{dinov2_encoder}.pth', map_location='cpu'), strict=True)
        if freeze:
            for param in self.depth_anything.pretrained.parameters():
                param.requires_grad = False
            for param in self.depth_anything.depth_head.parameters():
                param.requires_grad = False
        
        self.out_dim = model_configs[self.dinov2_encoder]['features']

    def forward(self, x, danv2_io_sizes, return_idepth=False):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError("DefomEncoder expects [left, right] image tensors")
        batch_size = x[0].shape[0]
        if x[1].shape[0] != batch_size:
            raise ValueError("left/right image batches must have equal size")
        x = torch.cat(x, dim=0)
        ih, iw, oh, ow = danv2_io_sizes
        x = F.interpolate(x, (ih, iw), mode="bilinear", align_corners=True)

        if return_idepth:
            features, left_feat, right_feat, idepth = self.depth_anything.forward_test(x, oh, ow)
            bs = idepth.shape[0]
            max_idepth, _ = torch.max(idepth.view(bs, -1), dim=1)
            max_idepth = max_idepth.detach().view(bs, 1, 1, 1) + 1e-8
            idepth = idepth / max_idepth * self.idepth_scale * ow + 0.01
            if idepth.shape[0] == batch_size:
                # This checkout's DPTHead already evaluates only the left
                # half of the concatenated stereo batch.
                left_idepth = idepth.contiguous()
            elif idepth.shape[0] == 2 * batch_size:
                # Keep compatibility with DepthAnything heads that return
                # both views explicitly.
                left_idepth = idepth[:batch_size].contiguous()
            else:
                raise RuntimeError(
                    "DepthAnythingV2 inverse-depth batch contract changed: "
                    f"got {idepth.shape[0]}, expected {batch_size} or "
                    f"{2 * batch_size}"
                )
            # Stereo predicts left disparity.  The right-image monocular map is
            # useful to the shared feature backbone but must never enter the
            # left recurrent state as an extra batch element.
            return features, left_feat, right_feat, left_idepth, None

        features, left_feat, right_feat = self.depth_anything(x, oh, ow)
        return features, left_feat, right_feat

class MultiBasicEncoder2(nn.Module):
    def __init__(self, output_dim=[128], norm_fn='batch', dropout=0.0, downsample=3):
        super(MultiBasicEncoder2, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)

        output_list = []
        
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs04 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, dual_inp=False, num_layers=3):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if dual_inp:
            v = x
            x = x[:(x.shape[0]//2)]

        outputs04 = [f(x) for f in self.outputs04]
        if num_layers == 1:
            return (outputs04, v) if dual_inp else (outputs04,)

        y = self.layer4(x)
        outputs08 = [f(y) for f in self.outputs08]

        if num_layers == 2:
            return (outputs04, outputs08, v) if dual_inp else (outputs04, outputs08)

        z = self.layer5(y)
        outputs16 = [f(z) for f in self.outputs16]

        return (outputs04, outputs08, outputs16, v) if dual_inp else (outputs04, outputs08, outputs16)