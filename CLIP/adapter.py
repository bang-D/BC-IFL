import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
# from pytorch_wavelets import DWTForward, DWTInverse
from torch.nn import init
import torchvision


class ClipAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(ClipAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        x = self.fc1(x)
        y = self.fc2(x)
        return x, y


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 初始化权重和偏置
        self.train_weight = nn.Parameter(torch.empty((in_features * 2, out_features)), requires_grad=True)  # 权重矩阵
        init.kaiming_normal_(self.train_weight.data, a=0, mode='fan_out')

    def forward(self, x, mode):
        if mode == 'train':
            return x @ self.train_weight


class DBCP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 生成一个长度为50的从0到pi的均匀分布的输入
        self.weight = torch.linspace(0, torch.pi, 20)

        # 计算余弦序列
        self.weight = torch.cos(self.weight)

        # 缩放并偏移余弦序列，使其值从0.99下降到0.01
        self.weight = 0.99 - 0.98 * (self.weight - self.weight.min()) / (self.weight.max() - self.weight.min())

        self.out_channels = out_channels

        self.linear4_inner = MyLinear(in_channels, out_channels)
        self.linear3_inner = MyLinear(in_channels, out_channels)
        self.linear2_inner = MyLinear(in_channels, out_channels)
        self.linear1_inner = MyLinear(in_channels, out_channels)

        self.linear_fuse_inner = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 4, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.linear_pred_inner = nn.Conv2d(out_channels, 1, kernel_size=1)

        self.dropout_inner = nn.Dropout2d(0.1)

        self.global_inner_pre = MyLinear(512, 768)
        self.global_inner_linear = MyLinear(in_channels, 1)

        self.linear4_outer = MyLinear(in_channels, out_channels)
        self.linear3_outer = MyLinear(in_channels, out_channels)
        self.linear2_outer = MyLinear(in_channels, out_channels)
        self.linear1_outer = MyLinear(in_channels, out_channels)

        self.linear_fuse_outer = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 4, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.linear_pred_outer = nn.Conv2d(out_channels, 1, kernel_size=1)

        self.dropout_outer = nn.Dropout2d(0.1)

        self.global_outer_linear = MyLinear(in_channels, 1)
        self.global_outer_pre = MyLinear(512, 768)

    def forward(self, inner_features, outer_features, mode, epoch, inner_tokens, outer_tokens):

        if mode == 'train':
            c1_inner, c2_inner, c3_inner, c4_inner = [inner_features[idx].detach() for idx in range(len(inner_features))]
            c1_outer, c2_outer, c3_outer, c4_outer = [outer_features[idx].detach() for idx in range(len(outer_features))]
            # print(c1_inner.shape, c2_inner.shape, c3_inner.shape, c4_inner.shape)
            # print(c1_outer.shape, c2_outer.shape, c3_outer.shape, c4_outer.shape)
        else:
            c1, c2, c3, c4 = inner_features

        B, L, C = c4_inner.shape
        H = int(np.sqrt(L))
        c4_inner = self.linear4_inner(c4_inner, mode).permute(0, 2, 1).reshape(B, self.out_channels, H, H)
        c3_inner = self.linear3_inner(c3_inner, mode).permute(0, 2, 1).reshape(B, self.out_channels, H, H)
        c2_inner = self.linear2_inner(c2_inner, mode).permute(0, 2, 1).reshape(B, self.out_channels, H, H)
        c1_inner = self.linear1_inner(c1_inner, mode).permute(0, 2, 1).reshape(B, self.out_channels, H, H)

        c_inner = torch.cat([c4_inner, c3_inner, c2_inner, c1_inner], dim=1)

        c_inner = self.linear_fuse_inner(c_inner)
        c_inner = self.dropout_inner(c_inner)
        best_th_inner = self.linear_pred_inner(c_inner)
        inner_tokens = self.global_inner_pre(inner_tokens, mode)
        best_th_inner = F.interpolate(best_th_inner, size=(512, 512), mode='bilinear', align_corners=True) + self.global_inner_linear(inner_tokens, mode)
        # best_th_inner = torch.sigmoid(best_th_inner)

        # ==============================================================================================================
        B, L, C = c4_outer.shape
        H = int(np.sqrt(L))
        c4_outer = self.linear4_outer(c4_outer, mode).permute(0, 2, 1).reshape(B, self.out_channels, H, H)
        c3_outer = self.linear3_outer(c3_outer, mode).permute(0, 2, 1).reshape(B, self.out_channels, H, H)
        c2_outer = self.linear2_outer(c2_outer, mode).permute(0, 2, 1).reshape(B, self.out_channels, H, H)
        c1_outer = self.linear1_outer(c1_outer, mode).permute(0, 2, 1).reshape(B, self.out_channels, H, H)

        c_outer = torch.cat([c4_outer, c3_outer, c2_outer, c1_outer], dim=1)

        c_outer = self.linear_fuse_outer(c_outer)
        c_outer = self.dropout_outer(c_outer)
        best_th_outer = self.linear_pred_outer(c_outer)
        outer_tokens = self.global_outer_pre(outer_tokens, mode)
        best_th_outer = F.interpolate(best_th_outer, size=(512, 512), mode='bilinear', align_corners=True) + self.global_outer_linear(outer_tokens, mode)
        # best_th_outer = torch.sigmoid(best_th_outer)

        if epoch < 20:
            best_th = self.weight[epoch].item() * best_th_inner + (1 - self.weight[epoch].item()) * best_th_outer
        else:
            best_th = self.weight[-1].item() * best_th_inner + (1 - self.weight[-1].item()) * best_th_outer

        best_th = torch.sigmoid(best_th)
        return best_th, inner_tokens, outer_tokens


class FFN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.linear4 = MyLinear(in_channels, out_channels)
        self.linear3 = MyLinear(in_channels, out_channels)
        self.linear2 = MyLinear(in_channels, out_channels)
        self.linear1 = MyLinear(in_channels, out_channels)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 4, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.linear_pred = nn.Conv2d(out_channels, 1, kernel_size=1)

        self.dropout = nn.Dropout2d(0.1)

    def forward(self, image_features, mode):
        if mode == 'train':
            c1, c2, c3, c4 = image_features
        else:
            c1, c2, c3, c4 = image_features
        B, L, C = c4.shape
        H = int(np.sqrt(L))
        c4 = self.linear4(c4, mode).permute(0, 2, 1).reshape(B, self.out_channels, H, H)
        c3 = self.linear3(c3, mode).permute(0, 2, 1).reshape(B, self.out_channels, H, H)
        c2 = self.linear2(c2, mode).permute(0, 2, 1).reshape(B, self.out_channels, H, H)
        c1 = self.linear1(c1, mode).permute(0, 2, 1).reshape(B, self.out_channels, H, H)

        c = torch.cat([c4, c3, c2, c1], dim=1)

        c = self.linear_fuse(c)
        c = self.dropout(c)
        output = self.linear_pred(c)
        output = F.interpolate(output, size=(512, 512), mode='bilinear', align_corners=True)
        output = torch.sigmoid(output)
        return output


class BCIFL(nn.Module):
    def __init__(self, clip_model, features):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.features = features
        self.tamper_adapters = nn.ModuleList([ClipAdapter(1024, bottleneck=768) for i in range(len(features))])
        self.bias_adapters = nn.ModuleList([ClipAdapter(1024, bottleneck=768) for i in range(len(features))])

        self.dbcp = DBCP(384, 256)

        self.bias_ffn = FFN(384, 256)
        self.tamper_ffn = FFN(384, 256)

    def forward(self, x, text_feature_m, text_feature_d, mode, epoch):
        with torch.no_grad():
            x = self.image_encoder.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)

            x = torch.cat(
                [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                 x], dim=1)
            x = x + self.image_encoder.positional_embedding.to(x.dtype)

            x = self.image_encoder.patch_dropout(x)
            x = self.image_encoder.ln_pre(x)

            x = x.permute(1, 0, 2)

        tamper_feature_list, bias_feature_list = [], []

        for i in range(24):
            with torch.no_grad():
                if i == 0:
                    clip_feature, _ = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
                else:
                    clip_feature, _ = self.image_encoder.transformer.resblocks[i](clip_feature, attn_mask=None)

            if (i + 1) in self.features:
                tamper_feature, tamper_weight = self.tamper_adapters[self.features.index(i+1)](clip_feature)

                clip_feature = 0.9 * clip_feature + 0.1 * tamper_weight

                tamper_feature_list.append(tamper_feature)

        tamper_cls_token = clip_feature.permute(1, 0, 2)[:, 0, :]

        for i in range(24):
            with torch.no_grad():
                if i == 0:
                    clip_feature, _ = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
                else:
                    clip_feature, _ = self.image_encoder.transformer.resblocks[i](clip_feature, attn_mask=None)

            if (i + 1) in self.features:
                bias_feature, bias_weight = self.bias_adapters[self.features.index(i+1)](clip_feature)

                clip_feature = 0.9 * clip_feature + 0.1 * bias_weight

                bias_feature_list.append(bias_feature)

        bias_cls_token = clip_feature.permute(1, 0, 2)[:, 0, :]

        tamper_feature_list = [tamper_feature_list[t].permute(1, 0, 2) for t in range(len(tamper_feature_list))]
        tamper_feature_list = [p[:, 1:, :] for p in tamper_feature_list]
        bias_feature_list = [bias_feature_list[t].permute(1, 0, 2) for t in range(len(bias_feature_list))]
        bias_feature_list = [p[:, 1:, :] for p in bias_feature_list]

        pixel_th, bias_cls_token, tamper_cls_token = self.dbcp(bias_feature_list, tamper_feature_list, mode, epoch, bias_cls_token, tamper_cls_token)

        bias_prediction = self.bias_ffn(bias_feature_list, mode)
        tamper_prediction = self.tamper_ffn(tamper_feature_list, mode)

        return bias_prediction, tamper_prediction, pixel_th, bias_cls_token, tamper_cls_token
