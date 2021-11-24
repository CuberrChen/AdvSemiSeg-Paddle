# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.utils import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models.layers.layer_libs import SyncBatchNorm


@manager.MODELS.add_component
class DualResNet(nn.Layer):

    def __init__(self,
                 num_classes,
                 block,
                 layers,
                 planes=64,
                 spp_planes=128,
                 head_planes=128,
                 augment=False,
                 align_corners=False,
                 pretrained=None):
        super(DualResNet, self).__init__()

        highres_planes = planes * 2
        self.augment = augment

        self.conv1 = nn.Sequential(
            nn.Conv2D(3, planes, kernel_size=3, stride=2, padding=1),
            SyncBatchNorm(planes, momentum=0.1),
            nn.ReLU(),
            nn.Conv2D(planes, planes, kernel_size=3, stride=2, padding=1),
            SyncBatchNorm(planes, momentum=0.1),
            nn.ReLU(),
        )

        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, planes, planes, layers[0])
        self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes * 2, planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes * 4, planes * 8, layers[3], stride=2)

        self.compression3 = nn.Sequential(
            nn.Conv2D(planes * 4, highres_planes, kernel_size=1, bias_attr=False),
            SyncBatchNorm(highres_planes, momentum=0.1),
        )

        self.compression4 = nn.Sequential(
            nn.Conv2D(planes * 8, highres_planes, kernel_size=1, bias_attr=False),
            SyncBatchNorm(highres_planes, momentum=0.1),
        )

        self.down3 = nn.Sequential(
            nn.Conv2D(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias_attr=False),
            SyncBatchNorm(planes * 4, momentum=0.1),
        )

        self.down4 = nn.Sequential(
            nn.Conv2D(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias_attr=False),
            SyncBatchNorm(planes * 4, momentum=0.1),
            nn.ReLU(),
            nn.Conv2D(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias_attr=False),
            SyncBatchNorm(planes * 8, momentum=0.1),
        )

        self.layer3_ = self._make_layer(block, planes * 2, highres_planes, 2)

        self.layer4_ = self._make_layer(block, highres_planes, highres_planes, 2)

        self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1)

        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)

        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)

        if self.augment:
            self.seghead_extra = segmenthead(highres_planes, head_planes, num_classes)

        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):

        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        ori_shape = x.shape[2:]
        layers = []

        x = self.conv1(x)

        x = self.layer1(x)
        layers.append(x)

        x = self.layer2(self.relu(x))
        layers.append(x)

        x = self.layer3(self.relu(x))
        layers.append(x)
        x_ = self.layer3_(self.relu(layers[1]))

        x = x + self.down3(self.relu(x_))
        x_ = x_ + F.interpolate(
            self.compression3(self.relu(layers[2])),
            size=[height_output, width_output],
            mode='bilinear')
        if self.augment:
            temp = x_

        x = self.layer4(self.relu(x))
        layers.append(x)
        x_ = self.layer4_(self.relu(x_))

        x = x + self.down4(self.relu(x_))
        x_ = x_ + F.interpolate(
            self.compression4(self.relu(layers[3])),
            size=[height_output, width_output],
            mode='bilinear')

        x_ = self.layer5_(self.relu(x_))
        x = F.interpolate(
            self.spp(self.layer5(self.relu(x))),
            size=[height_output, width_output],
            mode='bilinear')

        x_ = self.final_layer(x + x_)

        if self.augment:
            x_extra = self.seghead_extra(temp)
            logits = [x_, x_extra]
        else:
            logits = [x_]

        logit_list = [
            F.interpolate(
                logit,
                ori_shape,
                mode='bilinear',
                align_corners=self.align_corners) for logit in logits
        ]
        return logit_list

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                SyncBatchNorm(planes * block.expansion, momentum=0.1),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)
        else:
            for sublayer in self.sublayers():
                if isinstance(sublayer, nn.Conv2D):
                    param_init.kaiming_normal_init(sublayer.weight)
                elif isinstance(sublayer, (nn.BatchNorm, nn.SyncBatchNorm)):
                    param_init.constant_init(sublayer.weight, value=1.0)
                    param_init.constant_init(sublayer.bias, value=0.0)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias_attr=False)
        self.bn1 = SyncBatchNorm(planes, momentum=0.1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(planes, planes,kernel_size=3, stride=1,
                     padding=1, bias_attr=False)
        self.bn2 = SyncBatchNorm(planes, momentum=0.1)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)


class Bottleneck(nn.Layer):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = SyncBatchNorm(planes, momentum=0.1)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias_attr=False)
        self.bn2 = SyncBatchNorm(planes, momentum=0.1)
        self.conv3 = nn.Conv2D(planes, planes * self.expansion, kernel_size=1,
                               bias_attr=False)
        self.bn3 = SyncBatchNorm(planes * self.expansion, momentum=0.1)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class DAPPM(nn.Layer):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2D(kernel_size=5, stride=2, padding=2),
                                    SyncBatchNorm(inplanes, momentum=0.1),
                                    nn.ReLU(),
                                    nn.Conv2D(inplanes, branch_planes, kernel_size=1, bias_attr=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2D(kernel_size=9, stride=4, padding=4),
                                    SyncBatchNorm(inplanes, momentum=0.1),
                                    nn.ReLU(),
                                    nn.Conv2D(inplanes, branch_planes, kernel_size=1, bias_attr=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2D(kernel_size=17, stride=8, padding=8),
                                    SyncBatchNorm(inplanes, momentum=0.1),
                                    nn.ReLU(),
                                    nn.Conv2D(inplanes, branch_planes, kernel_size=1, bias_attr=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2D((1, 1)),
                                    SyncBatchNorm(inplanes, momentum=0.1),
                                    nn.ReLU(),
                                    nn.Conv2D(inplanes, branch_planes, kernel_size=1, bias_attr=False),
                                    )
        self.scale0 = nn.Sequential(
            SyncBatchNorm(inplanes, momentum=0.1),
            nn.ReLU(),
            nn.Conv2D(inplanes, branch_planes, kernel_size=1, bias_attr=False),
        )
        self.process1 = nn.Sequential(
            SyncBatchNorm(branch_planes, momentum=0.1),
            nn.ReLU(),
            nn.Conv2D(branch_planes, branch_planes, kernel_size=3, padding=1, bias_attr=False),
        )
        self.process2 = nn.Sequential(
            SyncBatchNorm(branch_planes, momentum=0.1),
            nn.ReLU(),
            nn.Conv2D(branch_planes, branch_planes, kernel_size=3, padding=1, bias_attr=False),
        )
        self.process3 = nn.Sequential(
            SyncBatchNorm(branch_planes, momentum=0.1),
            nn.ReLU(),
            nn.Conv2D(branch_planes, branch_planes, kernel_size=3, padding=1, bias_attr=False),
        )
        self.process4 = nn.Sequential(
            SyncBatchNorm(branch_planes, momentum=0.1),
            nn.ReLU(),
            nn.Conv2D(branch_planes, branch_planes, kernel_size=3, padding=1, bias_attr=False),
        )
        self.compression = nn.Sequential(
            SyncBatchNorm(branch_planes * 5, momentum=0.1),
            nn.ReLU(),
            nn.Conv2D(branch_planes * 5, outplanes, kernel_size=1, bias_attr=False),
        )
        self.shortcut = nn.Sequential(
            SyncBatchNorm(inplanes, momentum=0.1),
            nn.ReLU(),
            nn.Conv2D(inplanes, outplanes, kernel_size=1, bias_attr=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                                                    size=[height, width],
                                                    mode='bilinear') + x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[3])))

        out = self.compression(paddle.concat(x_list, 1)) + self.shortcut(x)
        return out


class segmenthead(nn.Layer):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = SyncBatchNorm(inplanes, momentum=0.1)
        self.conv1 = nn.Conv2D(inplanes, interplanes, kernel_size=3, padding=1, bias_attr=False)
        self.bn2 = SyncBatchNorm(interplanes, momentum=0.1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(interplanes, outplanes, kernel_size=1, padding=0, bias_attr=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                                size=[height, width],
                                mode='bilinear')

        return out


@manager.MODELS.add_component
def DDRNet23_slim(**kwargs):
    return DualResNet(block=BasicBlock,layers=[2, 2, 2, 2],
                      planes=32, spp_planes=128,
                      head_planes=64,**kwargs)


@manager.MODELS.add_component
def DDRNet23(**kwargs):
    return DualResNet(block=BasicBlock,layers=[2, 2, 2, 2],
                      planes=64, spp_planes=128,
                      head_planes=128,**kwargs)


@manager.MODELS.add_component
def DDRNet39(**kwargs):
    return DualResNet(block=BasicBlock,layers=[3, 4, 6, 3],
                      planes=64, spp_planes=128,
                      head_planes=256,**kwargs)