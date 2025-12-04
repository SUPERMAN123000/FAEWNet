# # Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from opencd.registry import MODELS
# ####################################################
from einops import rearrange

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class FeatureReuseModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x, reused_features):
        return self.conv(x) + reused_features  # 特征重用

class EnsembleLearningModule(nn.Module):
    def __init__(self, in_channels, num_models=3):
        super().__init__()
        self.models = nn.ModuleList([nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1) for _ in range(num_models)])

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)  # 集成学习

class CDCM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CDCM, self).__init__()

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False)
        self.conv2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False)
        self.conv2_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=11, padding=11, bias=False)
        self.DepthwiseSeparableConv = DepthwiseSeparableConv(out_channels, out_channels)
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        x = self.relu1(x)
        x = self.conv1(x)
        x1 = self.DepthwiseSeparableConv(self.conv2_1(x))
        x2 = self.DepthwiseSeparableConv(self.conv2_2(x))
        x3 = self.DepthwiseSeparableConv(self.conv2_3(x))
        x4 = self.DepthwiseSeparableConv(self.conv2_4(x))

        return x1 + x2 + x3 + x4

class MSAI(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size), padding=kernel_size // 2)
        self.get_weights = nn.Sequential(
            nn.Conv2d(in_channel * (kernel_size ** 2), in_channel * (kernel_size ** 2), kernel_size=1,
                      groups=in_channel),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)))

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=0, stride=kernel_size)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU()
        self.cdcm = CDCM(in_channel, in_channel)

        self.feature_reuse = FeatureReuseModule(in_channel)
        self.ensemble_learning = EnsembleLearningModule(in_channel)


    def forward(self, x):
        x = self.cdcm(x)
        b, c, h, w = x.shape
        unfold_feature = self.unfold(x)  # 获得感受野空间特征  b c*kernel**2,h*w
        x = unfold_feature
        data = unfold_feature.unsqueeze(-1)
        weight = self.get_weights(data).view(b, c, self.kernel_size ** 2, h, w).permute(0, 1, 3, 4, 2).softmax(-1)
        weight_out = rearrange(weight, 'b c h w (n1 n2) -> b c (h n1) (w n2)', n1=self.kernel_size, n2=self.kernel_size) # b c h w k**2 -> b c h*k w*k
        receptive_field_data = rearrange(x, 'b (c n1) l -> b c n1 l', n1=self.kernel_size ** 2).permute(0, 1, 3, 2).reshape(b, c, h, w, self.kernel_size ** 2) # b c*kernel**2,h*w ->  b c h w k**2
        data_out = rearrange(receptive_field_data, 'b c h w (n1 n2) -> b c (h n1) (w n2)', n1=self.kernel_size,n2=self.kernel_size) # b c h w k**2 -> b c h*k w*k
        conv_data = data_out * weight_out
        conv_out = self.conv(conv_data)

        reused_features = conv_out  # 作为重用特征
        output = self.feature_reuse(conv_out, reused_features)
        output = self.ensemble_learning(output)
        return self.act(self.bn(output))

class FDAF(BaseModule):
    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='IN'),
                 act_cfg=dict(type='GELU')):
        super(FDAF, self).__init__()
        self.in_channels = in_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        # TODO
        conv_cfg = None
        norm_cfg = dict(type='IN')
        act_cfg = dict(type='GELU')

        kernel_size = 5
        self.flow_make = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                      bias=True, groups=in_channels * 2),
            nn.InstanceNorm2d(in_channels * 2),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, 4, kernel_size=1, padding=0, bias=False)
        )
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.RFAConv = MSAI(in_channels,in_channels)


    def forward(self, x1, x2):
        """Forward function."""
        rfa1 = self.RFAConv(x1)
        rfa2 = self.RFAConv(x2)
        output = torch.cat([x1, x2], dim=1)
        flow = self.flow_make(output)
        f1, f2 = torch.chunk(flow, 2, dim=1)
        x1_feat = self.warp(rfa1, f1) - x2
        x2_feat = self.warp(rfa2, f2) - x1

        return x1_feat, x2_feat

    @staticmethod
    def warp(x, flow):
        n, c, h, w = x.size()

        norm = torch.tensor([[[[w, h]]]]).type_as(x).to(x.device)
        col = torch.linspace(-1.0, 1.0, h).view(-1, 1).repeat(1, w)
        row = torch.linspace(-1.0, 1.0, w).repeat(h, 1)
        grid = torch.cat((row.unsqueeze(2), col.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(x).to(x.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(x, grid, align_corners=True)
        return output

@MODELS.register_module()
class MSAFANeck(BaseModule):
    """Feature Fusion Neck.

    Args:
        policy (str): The operation to fuse features. candidates
            are `concat`, `sum`, `diff` and `Lp_distance`.
        in_channels (Sequence(int)): Input channels.
        channels (int): Channels after modules, before conv_seg.
        out_indices (tuple[int]): Output from which layer.
    """

    def __init__(self,
                 policy,
                 in_channels=None,
                 channels=None,
                 out_indices=(0, 1, 2, 3)):
        super().__init__()
        self.policy = policy
        self.in_channels = in_channels
        self.channels = channels
        self.out_indices = out_indices
        self.FDAF = FDAF(in_channels=256)
    @staticmethod
    def fusion(x1, x2, policy):
        """Specify the form of feature fusion"""

        _fusion_policies = ['concat', 'sum', 'diff', 'abs_diff']
        assert policy in _fusion_policies, 'The fusion policies {} are ' \
            'supported'.format(_fusion_policies)

        if policy == 'concat':
            x = torch.cat([x1, x2], dim=1)
        elif policy == 'sum':
            x = x1 + x2
        elif policy == 'diff':
            x = x2 - x1
        elif policy == 'abs_diff':
            x = torch.abs(x1 - x2)

        return x

    def forward(self, x1, x2):
        """Forward function."""

        assert len(x1) == len(x2), "The features x1 and x2 from the" \
            "backbone should be of equal length"
        outs = []
        for i in range(len(x1)):
            x1[i], x2[i] = self.FDAF(x1[i],x2[i])
            out = self.fusion(x1[i], x2[i], self.policy)
            outs.append(out)

        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)