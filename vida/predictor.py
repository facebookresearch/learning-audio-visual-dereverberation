# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class Predictor(nn.Module):
    def __init__(self, input_channel=2, use_rgb=False, use_depth=False,
                 no_mask=False, limited_fov=False, mean_pool_visual=False, use_rgbd=False):
        super(Predictor, self).__init__()
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.use_rgbd = use_rgbd
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.use_visual = use_rgb or use_depth or use_rgbd
        self.mean_pool_visual = mean_pool_visual

        if self.use_visual:
            if use_rgb:
                self.rgb_net = VisualNet(torchvision.models.resnet18(pretrained=True), 3)
            if use_depth:
                self.depth_net = VisualNet(torchvision.models.resnet18(pretrained=True), 1)
            if use_rgbd:
                self.rgbd_net = VisualNet(torchvision.models.resnet18(pretrained=True), 4)
            concat_size = 512 * sum([self.use_rgb, self.use_depth, self.use_rgbd])
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
            if self.mean_pool_visual:
                self.conv1x1 = create_conv(concat_size, 512, 1, 0)
            else:
                self.conv1x1 = create_conv(concat_size, 8, 1, 0)  # reduce dimension of extracted visual features

        # if complex, keep input output channel as 2
        self.audio_net = AudioNet(64, input_channel, input_channel, self.use_visual, no_mask,
                                  limited_fov, mean_pool_visual)
        self.audio_net.apply(weights_init)

    def forward(self, inputs):
        visual_features = []
        if self.use_rgb:
            visual_features.append(self.rgb_net(inputs['rgb']))
        if self.use_depth:
            visual_features.append(self.depth_net(inputs['depth']))
        if self.use_rgbd:
            visual_features.append(self.rgbd_net(torch.cat([inputs['rgb'], inputs['depth']], dim=1)))
        if len(visual_features) != 0:
            # concatenate channel-wise
            concat_visual_features = torch.cat(visual_features, dim=1)
            concat_visual_features = self.conv1x1(concat_visual_features)
        else:
            concat_visual_features = None

        if self.mean_pool_visual:
            concat_visual_features = self.pooling(concat_visual_features)
        elif len(visual_features) != 0:
            concat_visual_features = concat_visual_features.view(concat_visual_features.shape[0], -1, 1, 1)

        pred_mask, audio_feat = self.audio_net(inputs['spectrograms'], concat_visual_features)
        output = {'pred_mask': pred_mask}

        if len(visual_features) != 0:
            audio_embed = self.pooling(audio_feat).squeeze(-1).squeeze(-1)
            visual_embed = concat_visual_features.squeeze(-1).squeeze(-1)
            output['audio_feat'] = F.normalize(audio_embed, p=2, dim=1)
            output['visual_feat'] = F.normalize(visual_embed, p=2, dim=1)

        return output


def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])


def unet_upconv(input_nc, output_nc, outermost=False, use_sigmoid=False, use_tanh=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        if use_sigmoid:
            return nn.Sequential(*[upconv, nn.Sigmoid()])
        elif use_tanh:
            return nn.Sequential(*[upconv, nn.Tanh()])
        else:
            return nn.Sequential(*[upconv])


def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, use_relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride=stride, padding=paddings)]
    if batch_norm:
        model.append(nn.BatchNorm2d(output_channels))
    if use_relu:
        model.append(nn.ReLU())
    return nn.Sequential(*model)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)


class VisualNet(nn.Module):
    def __init__(self, original_resnet, num_channel=3):
        super(VisualNet, self).__init__()
        original_resnet.conv1 = nn.Conv2d(num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        layers = list(original_resnet.children())[0:-2]
        self.feature_extraction = nn.Sequential(*layers)  # features before conv1x1

    def forward(self, x):
        x = self.feature_extraction(x)
        return x


class AudioNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2, use_visual=True, no_mask=False, limited_fov=False, 
                 mean_pool_visual=False):
        super(AudioNet, self).__init__()
        self.use_visual = use_visual
        self.no_mask = no_mask

        num_intermediate_feature = 512
        if use_visual:
            if limited_fov:
                num_intermediate_feature += (768 if not mean_pool_visual else 512)
            else:
                num_intermediate_feature += (864 if not mean_pool_visual else 512)

        # initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_upconvlayer1 = unet_upconv(num_intermediate_feature, ngf * 8) 
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 4)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        if self.no_mask:
            self.audionet_upconvlayer5 = unet_upconv(ngf * 2, output_nc, outermost=True)
        else:
            # outermost layer use a sigmoid to bound the mask
            self.audionet_upconvlayer5 = unet_upconv(ngf * 2, output_nc, outermost=True, use_sigmoid=True)

    def forward(self, x, visual_feat):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)

        upconv_feature_input = audio_conv5feature
        if self.use_visual:
            visual_feat = visual_feat.view(visual_feat.shape[0], -1, 1, 1)  # flatten visual feature
            visual_feat = visual_feat.repeat(1, 1, audio_conv5feature.shape[-2],
                                             audio_conv5feature.shape[-1])  # tile visual feature
            upconv_feature_input = torch.cat((visual_feat, upconv_feature_input), dim=1)

        audio_upconv1feature = self.audionet_upconvlayer1(upconv_feature_input)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv4feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv3feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv2feature), dim=1))
        if self.no_mask:
            prediction = self.audionet_upconvlayer5(
                torch.cat((audio_upconv4feature, audio_conv1feature), dim=1))
        else:
            prediction = self.audionet_upconvlayer5(
                torch.cat((audio_upconv4feature, audio_conv1feature), dim=1)) * 2 - 1

        return prediction, audio_conv5feature
