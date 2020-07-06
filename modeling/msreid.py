import torch
from torch import nn
from torch.nn import functional as F

from .backbones.resnet import ResNet, Bottleneck
from modeling.backbones.utils import weights_init_kaiming, weights_init_classifier


class ChanReduct(nn.Module):
    def __init__(self, in_chan, mid_chan, out_chan, bias=False, act=F.relu):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_chan, mid_chan, 1, bias=bias),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(),
            nn.Conv2d(mid_chan, mid_chan, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(),
            nn.Conv2d(mid_chan, out_chan, 1, bias=bias),
            nn.BatchNorm2d(out_chan),
        )
        self.bottleneck.apply(weights_init_kaiming)
        self.branch = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 1, bias=bias),
            nn.BatchNorm2d(out_chan),
        )
        self.branch.apply(weights_init_kaiming)
        self.act = act

    def forward(self, x):
        return self.act(self.branch(x) + self.bottleneck(x))


class SpatialAttn(nn.Module):
    def __init__(self, in_chan):
        super().__init__()
        self.conv0 = nn.Conv2d(in_chan, 1, kernel_size=1, bias=False)
        self.conv0.apply(weights_init_kaiming)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.conv1.apply(weights_init_kaiming)

        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.conv2.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = F.interpolate(x, (x.size(2) * 2, x.size(3) * 2), mode='bilinear', align_corners=True)
        x = self.conv2(x)

        return x


class ChanAttn(nn.Module):
    def __init__(self, in_chan):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, in_chan // 16, 1, bias=False),
            nn.BatchNorm2d(in_chan // 16),
            nn.ReLU(),
            nn.Conv2d(in_chan // 16, in_chan, 1, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.ReLU()
        )
        self.conv.apply(weights_init_kaiming)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.conv(x)

        return x


class SoftAttn(nn.Module):
    def __init__(self, in_chan):
        super().__init__()
        self.spatial_attn = SpatialAttn(in_chan)
        self.channel_attn = ChanAttn(in_chan)
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, in_chan, 1, bias=False),
            nn.BatchNorm2d(in_chan)
        )
        self.conv.apply(weights_init_kaiming)

    def forward(self, x):
        spatial = self.spatial_attn(x)
        channel = self.channel_attn(x)
        x = torch.sigmoid(self.conv(spatial * channel))

        return x


class MSReID(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, featmap_dim, model_path, model_name, use_attn, use_mask, use_local):
        super(MSReID, self).__init__()
        if model_name == 'resnet50':
            base = ResNet(last_stride=1,
                          block=Bottleneck,
                          layers=[3, 4, 6, 3])
        else:
            raise NotImplementedError()

        if model_path != '':
            base.load_param(model_path, strict=True)
            print(f'Loading pretrained model form {model_path}......')

        self.input_layer = nn.Sequential(
            nn.Sequential(
                base.conv1,
                base.bn1,
                base.relu,
                base.maxpool
            )
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.use_attn = use_attn
        if use_attn:
            self.soft_attn = [
                SoftAttn(self.in_planes // 8),
                SoftAttn(self.in_planes // 4),
                SoftAttn(self.in_planes // 2),
                SoftAttn(self.in_planes)
            ]
            self.soft_attn = nn.ModuleList(self.soft_attn)

        self.use_mask = use_mask
        if self.use_mask:
            self.mask = ChanReduct(self.in_planes, self.in_planes // 4, 1, bias=True, act=torch.sigmoid)

        self.reduction = ChanReduct(self.in_planes, self.in_planes // 4, featmap_dim, bias=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.num_classes = num_classes

        self.bottleneck = nn.BatchNorm1d(featmap_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(featmap_dim, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.use_local = use_local
        if self.use_local:
            local_chan = self.in_planes // 2
            self.local_reduction = ChanReduct(local_chan, local_chan // 4, featmap_dim, bias=True)

            self.local_bottleneck = nn.BatchNorm1d(featmap_dim)
            self.local_bottleneck.bias.requires_grad_(False)
            self.local_bottleneck.apply(weights_init_kaiming)

            self.local_classifier = nn.Linear(featmap_dim, self.num_classes, bias=False)
            self.local_classifier.apply(weights_init_classifier)

    def forward(self, x, labels=None):
        B = x.size(0)

        x = self.input_layer(x)

        x = self.layer1(x)
        if self.use_attn:
            x = self.soft_attn[0](x) * x

        x = self.layer2(x)
        if self.use_attn:
            x = self.soft_attn[1](x) * x

        x = self.layer3(x)
        if self.use_attn:
            x = self.soft_attn[2](x) * x

        if self.use_local:
            local_feat = self.local_reduction(x)
        else:
            local_feat = None

        x = self.layer4(x)
        if self.use_attn:
            x = self.soft_attn[3](x) * x

        if self.use_mask:
            global_mask = self.mask(x)
            local_mask = 1 - global_mask
        else:
            global_mask = torch.ones_like(x).mean(dim=1, keepdim=True)
            local_mask = torch.ones_like(x).mean(dim=1, keepdim=True)

        # triplet features
        global_feat = self.reduction(x)

        global_feat = global_feat * global_mask
        global_feat = self.gap(global_feat) + self.gmp(global_feat)
        global_feat = global_feat.view(B, -1)

        global_bn_feat = self.bottleneck(global_feat)
        global_cls_score = self.classifier(global_bn_feat)

        if self.use_local:
            local_feat = local_mask * local_feat
            local_feat = self.gap(local_feat) + self.gmp(local_feat)
            local_feat = local_feat.view(B, -1)

            local_bn_feat = self.local_bottleneck(local_feat)
            local_cls_score = self.local_classifier(local_bn_feat)

            return [global_cls_score, local_cls_score], [global_feat, local_feat]
        else:
            return [global_cls_score], [global_feat]
