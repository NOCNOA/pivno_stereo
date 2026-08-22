import torch
import torch.nn as nn
import torch.nn.functional as F
from .galerkin import simple_attn
from . import register

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        self.norm1 = nn.InstanceNorm2d(planes)
        self.norm2 = nn.InstanceNorm2d(planes)
        if not stride == 1:
            self.norm3 = nn.InstanceNorm2d(planes)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, padding=0, stride=stride))

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder256(nn.Module):
    def __init__(self, output_dim=64, norm_fn='instance', dropout=0.0,
                 input_channels=1):
        super(BasicEncoder256, self).__init__()

        if input_channels < 1:
            raise ValueError(
                f"input_channels must be positive, got {input_channels}"
            )

        self.norm_fn = norm_fn
        self.norm1 = nn.InstanceNorm2d(64)
        self.repPad = nn.ReplicationPad2d(1)  # 因为kernel_size=1，所以需要填充1个像素
        # Learned downsampling preserves matching detail better than resizing
        # stereo images manually in the dataset pipeline.
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=3, stride=2, padding=0
        )
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=1)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

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

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = self.repPad(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x



class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=256, input_dim=256):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h
@register('piv_gt')
@register('stereo_gt')
class PIVNO(nn.Module):

    def __init__(self, width=128, blocks=16, iters=5, input_channels=1):
        super().__init__()
        self.iters = iters
        self.width = width
        self.input_channels = int(input_channels)
        self.snet = BasicEncoder256(
            output_dim=64,
            norm_fn='instance',
            dropout=0.,
            input_channels=self.input_channels,
        )

        self.conv0 = simple_attn(self.width, blocks)
        self.conv1 = simple_attn(self.width, blocks)
        self.disp_head = nn.Sequential(
            nn.Conv2d(self.width, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
        )
        self.gru = SepConvGRU(hidden_dim=128, input_dim=128)

    def query_rgb(self, feat):

        # Full-resolution validation produces a larger feature grid than the
        # training crop. After training, the second Galerkin projection can
        # overflow FP16 on otherwise finite inputs and poison every recurrent
        # prediction. Keep just these two sensitive attention blocks in FP32;
        # the encoder, PIVNO GRU, and the outer refinement remain autocastable.
        with torch.cuda.amp.autocast(enabled=False):
            x = self.conv0(feat.float(), 0)
            x = self.conv1(x, 1)

        return x



    def forward(self, img1, img2, return_imgfeature=False):
        """Estimate full-resolution horizontal disparity from a stereo pair."""
        if img1.ndim != 4 or img2.ndim != 4:
            raise ValueError(
                "PIVNO inputs must be [B,C,H,W], got "
                f"{tuple(img1.shape)} and {tuple(img2.shape)}"
            )
        if img1.shape != img2.shape:
            raise ValueError(
                f"PIVNO stereo inputs must match, got "
                f"{tuple(img1.shape)} and {tuple(img2.shape)}"
            )
        if img1.shape[1] != self.input_channels:
            raise ValueError(
                f"PIVNO expects {self.input_channels} input channels, "
                f"got {img1.shape[1]}"
            )
        output_size = img1.shape[-2:]
        smap1, smap2 = self.snet([img1, img2])
        smap = torch.cat((smap1, smap2), dim=1)
        smap = self.query_rgb(smap)
        hidden = smap.clone()
        disparity_predictions = []

        for _ in range(self.iters):
            hidden = self.gru(hidden, smap)
            disparity_low = self.disp_head(hidden)
            scale_x = output_size[1] / disparity_low.shape[-1]
            disparity = F.interpolate(
                disparity_low,
                size=output_size,
                mode='bilinear',
                align_corners=False,
            )
            # Convert feature-map pixel units to original-image pixel units.
            disparity_predictions.append(disparity * scale_x)
        if return_imgfeature==False:
            return disparity_predictions
        else:
            return disparity_predictions, smap1, smap2, disparity_low
