"""Dual-mode recurrent update used only by ``defom_pact_bilap_gru``."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.update import SepConvGRU


def _conv_block(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.GroupNorm(8, out_channels), nn.GELU())


class BiLapUpdateBlock(nn.Module):
    """Update two Laplace modes with shared weights and symmetric interaction."""

    def __init__(self, hidden_dim=128, corr_dim=81, aligned_dim=64, context_dim=128, separate_mode_gru=False, interaction=True, up_factor=4, max_disp_quarter=192.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.separate_mode_gru = bool(separate_mode_gru)
        self.interaction = bool(interaction)
        self.max_disp_quarter = float(max_disp_quarter)
        self.mode_initializer = nn.Sequential(_conv_block(hidden_dim + 4, hidden_dim), nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.Tanh())
        self.hidden_summary = nn.Sequential(_conv_block(3 * hidden_dim, hidden_dim), nn.Conv2d(hidden_dim, hidden_dim, 1))
        self.global16_gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=2 * hidden_dim)
        self.global8_gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=2 * hidden_dim)
        evidence_channels = corr_dim + aligned_dim + context_dim + 4
        self.evidence_encoder = nn.Sequential(_conv_block(evidence_channels, hidden_dim), _conv_block(hidden_dim, hidden_dim))
        self.mode_summary = nn.Sequential(_conv_block(hidden_dim, 64), nn.Conv2d(64, 64, 1))
        self.interaction_encoder = nn.Sequential(_conv_block(64 * 3 + 4, 64), nn.Conv2d(64, 64, 1))
        self.mode_input = nn.Sequential(_conv_block(hidden_dim + 64 + hidden_dim, hidden_dim), nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1))
        if self.separate_mode_gru:
            self.mode_grus = nn.ModuleList([SepConvGRU(hidden_dim=hidden_dim, input_dim=hidden_dim) for _ in range(2)])
        else:
            self.mode_gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=hidden_dim)
        self.parameter_head = nn.Sequential(_conv_block(2 * hidden_dim, hidden_dim), nn.Conv2d(hidden_dim, 3, 3, padding=1))
        self.mask_head = nn.Sequential(_conv_block(4 * hidden_dim, hidden_dim), nn.Conv2d(hidden_dim, 9 * up_factor * up_factor, 1))
        nn.init.zeros_(self.parameter_head[-1].weight)
        nn.init.zeros_(self.parameter_head[-1].bias)

    @staticmethod
    def _flatten_modes(value):
        batch, modes, channels, height, width = value.shape
        return value.reshape(batch * modes, channels, height, width)

    @staticmethod
    def _unflatten_modes(value, batch, modes):
        return value.reshape(batch, modes, value.shape[1], value.shape[2], value.shape[3])

    def symmetric_summary(self, mode_hidden, weights):
        mean_hidden = mode_hidden.mean(dim=1)
        max_hidden = mode_hidden.max(dim=1).values
        weighted_hidden = (mode_hidden * weights.unsqueeze(2)).sum(dim=1)
        return self.hidden_summary(torch.cat((mean_hidden, max_hidden, weighted_hidden), dim=1))

    def initialize_modes(self, global_quarter, means, log_scales, logits, mono_disp):
        weights = torch.softmax(logits.float(), dim=1)
        scale_norm = max(self.max_disp_quarter, 1.0)
        statistics = torch.stack((means / scale_norm, log_scales / 4.0, weights, (mono_disp - means) / scale_norm), dim=2)
        global_modes = global_quarter.unsqueeze(1).expand(-1, means.shape[1], -1, -1, -1)
        inputs = torch.cat((global_modes, statistics), dim=2)
        flat = self.mode_initializer(self._flatten_modes(inputs))
        return self._unflatten_modes(flat, means.shape[0], means.shape[1])

    def _interaction(self, mode_hidden, means, log_scales, weights):
        batch, modes = mode_hidden.shape[:2]
        if not self.interaction:
            return mode_hidden.new_zeros(batch, modes, 64, mode_hidden.shape[-2], mode_hidden.shape[-1])
        flat_summary = self.mode_summary(self._flatten_modes(mode_hidden))
        summary = self._unflatten_modes(flat_summary, batch, modes)
        total = summary.sum(dim=1, keepdim=True)
        other = total - summary
        max_summary = summary.max(dim=1, keepdim=True).values.expand_as(summary)
        separation = (means - means.flip(1)).abs().unsqueeze(2)
        log_scale_difference = (log_scales - log_scales.flip(1)).unsqueeze(2)
        own_weight = weights.unsqueeze(2)
        other_weight = weights.flip(1).unsqueeze(2)
        inputs = torch.cat((summary, other, max_summary, separation, log_scale_difference, own_weight, other_weight), dim=2)
        interaction = self.interaction_encoder(self._flatten_modes(inputs))
        return self._unflatten_modes(interaction, batch, modes)

    def forward(self, mode_hidden, global8, global16, corr, aligned, means, log_scales, logits, mono_disp, context):
        batch, modes, _, height, width = mode_hidden.shape                                      # 读取两个1/4 mode隐状态
        weights = torch.softmax(logits.float(), dim=1)                                          # 将混合logits转成每个mode的概率
        old_summary = self.symmetric_summary(mode_hidden, weights)                              # 对旧mode状态做置换不变汇总
        recurrent_summary = old_summary if self.interaction else torch.zeros_like(old_summary)  # 无交互消融不把mode摘要反馈到共享全局状态
        summary16 = F.interpolate(recurrent_summary, size=global16.shape[-2:], mode="bilinear", align_corners=True)  # 将mode摘要送到1/16尺度
        pooled8 = F.interpolate(global8, size=global16.shape[-2:], mode="bilinear", align_corners=True)       # 将1/8全局状态对齐到1/16
        global16 = self.global16_gru(global16, summary16, pooled8)                               # 更新共享1/16全局状态
        summary8 = F.interpolate(recurrent_summary, size=global8.shape[-2:], mode="bilinear", align_corners=True)   # 将mode摘要送到1/8尺度
        guide16 = F.interpolate(global16, size=global8.shape[-2:], mode="bilinear", align_corners=True)       # 上采样新1/16状态
        global8 = self.global8_gru(global8, summary8, guide16)                                   # 更新共享1/8全局状态
        mono_modes = mono_disp.expand(-1, modes, -1, -1)                                        # 为每个mode复制单目提示
        norm = max(self.max_disp_quarter, 1.0)                                                    # 使用固定最大视差进行数值归一化
        scalars = torch.stack((means / norm, log_scales / 4.0, weights, (mono_modes - means) / norm), dim=2)  # 组织mode标量证据
        context_modes = context.unsqueeze(1).expand(-1, modes, -1, -1, -1)                       # 为每个mode共享左图上下文
        evidence = torch.cat((corr, aligned, context_modes, scalars), dim=2)                      # 拼接匹配、对齐、上下文和分布证据
        evidence = self._unflatten_modes(self.evidence_encoder(self._flatten_modes(evidence)), batch, modes)  # 独立编码每个mode证据
        interaction = self._interaction(mode_hidden, means, log_scales, weights)                 # 计算对称的mode间交互
        global_guide = F.interpolate(global8, size=(height, width), mode="bilinear", align_corners=True)      # 将1/8全局状态对齐到1/4
        global_modes = global_guide.unsqueeze(1).expand(-1, modes, -1, -1, -1)                   # 两个mode使用同一全局引导
        mode_input = torch.cat((evidence, interaction, global_modes), dim=2)                      # 构造每个mode的GRU输入
        mode_input = self._unflatten_modes(self.mode_input(self._flatten_modes(mode_input)), batch, modes)    # 压缩到128维
        old_hidden = mode_hidden                                                                 # 保留同步更新所需的旧状态
        if self.separate_mode_gru:                                                               # 可选的非共享GRU消融
            updated = [self.mode_grus[index](old_hidden[:, index], mode_input[:, index]) for index in range(modes)]  # 分别更新各mode
            mode_hidden = torch.stack(updated, dim=1)                                            # 恢复[B,M,C,H,W]
        else:
            flat_hidden = self.mode_gru(self._flatten_modes(old_hidden), self._flatten_modes(mode_input))     # 用共享GRU同步更新所有mode
            mode_hidden = self._unflatten_modes(flat_hidden, batch, modes)                        # 恢复mode维度
        head_input = torch.cat((mode_hidden, mode_input), dim=2)                                  # 联合新隐状态和本轮证据
        raw_update = self.parameter_head(self._flatten_modes(head_input))                         # 预测mu、log_b和logit三种增量
        raw_update = self._unflatten_modes(raw_update, batch, modes)                              # 恢复每个mode的更新量
        up_mask = 0.25 * self.mask_head(torch.cat((mode_hidden.mean(dim=1), mode_hidden.max(dim=1).values, (mode_hidden * weights.unsqueeze(2)).sum(dim=1), global_guide), dim=1))  # 生成共享凸上采样mask
        return mode_hidden, global8, global16, raw_update, up_mask                                # 返回所有循环状态和参数更新


__all__ = ["BiLapUpdateBlock"]
