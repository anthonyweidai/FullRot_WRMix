from typing import Any

from torch import nn, Tensor

from .. import registerClsModels
from ..base import BaseEncoder
from ...misc import moduleProfile
from ...layers import BaseConv2d, AdaptiveAvgPool2d, Linearlayer, initWeight
from ....utils import setMethod


''' Reference to
https://github.com/facebookresearch/pycls (official has pretrained models)
https://github.com/d-li14/regnet.pytorch
https://github.com/signatrix/regnet/blob/master/src/modules.py
'''


class Bottleneck(nn.Module):
    expansion = 1
    __constants__ = ['Downsample']

    def __init__(self, inplanes, planes, stride=1, Downsample=None, group_width=1,
                 dilation=1, norm_layer=None, SERatio=None, **kwargs: Any):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = planes * self.expansion
        # Both self.conv2 and self.Downsample Layers Downsample the input when stride != 1
        self.Conv1 = BaseConv2d(inplanes, width, 1, BNorm=True, ActLayer=nn.ReLU)
        
        self.Downsample = Downsample
        
        self.Conv2 = BaseConv2d(
            width, width, 3, stride, groups=width // min(width, group_width), 
            dilation=dilation, BNorm=True, ActLayer=nn.ReLU
        )
        
        self.SERatio = SERatio
        if self.SERatio is not None:
            # SEChannels = width // self.SERatio
            SEChannels = int(round(inplanes / self.SERatio))
            self.SE = nn.Sequential(
                AdaptiveAvgPool2d(output_size=1),
                BaseConv2d(width, SEChannels, kernel_size=1, ActLayer=nn.ReLU),
                BaseConv2d(SEChannels, width, kernel_size=1),
                nn.Sigmoid(),
                )
        else:
            self.SE = None

        self.Conv3 = BaseConv2d(width, planes, 1, BNorm=True)

        self.Relu = nn.ReLU(inplace=True)
        
        self.stride = stride

    def forward(self, x):
        Identity = x

        Out = self.Conv1(x)
        Out = self.Conv2(Out)
        
        if self.SERatio:
            Out = Out * self.SE(Out)

        Out = self.Conv3(Out)

        if self.Downsample is not None:
            Identity = self.Downsample(x)

        Out += Identity
        Out = self.Relu(Out)

        return Out

    def profileModule(self, Input: Tensor):
        Params = MACs = 0.0
        
        Output, ParamsTemp, MACsTemp = moduleProfile(module=self.Conv1, x=Input)
        Params += ParamsTemp
        MACs += MACsTemp
        
        Output, ParamsTemp, MACsTemp = moduleProfile(module=self.Conv2, x=Output)
        Params += ParamsTemp
        MACs += MACsTemp
        
        if self.SERatio:
            _, ParamsTemp, MACsTemp = moduleProfile(module=self.SE, x=Output)
            Params += ParamsTemp
            MACs += MACsTemp

        Output, ParamsTemp, MACsTemp = moduleProfile(module=self.Conv3, x=Output)
        Params += ParamsTemp
        MACs += MACsTemp
        
        if self.Downsample is not None:
            _, ParamsTemp, MACsTemp = moduleProfile(module=self.Downsample, x=Input)
            Params += ParamsTemp
            MACs += MACsTemp
         
        return Output, Params, MACs


class RegNet(BaseEncoder):
    def __init__(self, block, Layers, widths, opt, 
                 group_width=1, replace_stride_with_dilation=None,
                 **kwargs: Any):
        super(RegNet, self).__init__(opt, **kwargs)
        self.inplanes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False, False]
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self.group_width = group_width
        
        InChannels = 3
        OutChannels = self.inplanes
        self.Conv1 = BaseConv2d(InChannels, OutChannels, kernel_size=3, stride=2, BNorm=True, ActLayer=nn.ReLU)
        self.ModelConfigDict['conv1'] = {'in': InChannels, 'out': OutChannels, 'stage': 1}

        for i in range(4):
            InChannels = OutChannels
            OutChannels = widths[i]
            Layer = self.makeLayer(block, widths[i], Layers[i], stride=2, 
                                   dilate=replace_stride_with_dilation[i], **kwargs)
            setMethod(self, 'Layer%d' % (i + 1), Layer)
            self.ModelConfigDict['layer%d' % (i + 1)] = {'in': InChannels, 'out': OutChannels, 'stage': i + 2}
        
        self.Classifier = Linearlayer(OutChannels * block.expansion, self.NumClasses)

        if opt.init_weight:
            self.apply(initWeight)

    def makeLayer(self, block, planes, blocks, stride=1, dilate=False, **kwargs: Any):
        Downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes:
            Downsample = BaseConv2d(self.inplanes, planes, 1, stride, BNorm=True)

        Layers = []
        Layers.append(block(self.inplanes, planes, stride, Downsample, self.group_width,
                            previous_dilation, **kwargs))
        self.inplanes = planes
        for _ in range(1, blocks):
            Layers.append(block(self.inplanes, planes, group_width=self.group_width,
                                dilation=self.dilation, **kwargs))

        return nn.Sequential(*Layers)

    def forwardTuple(self, x: Tensor) -> list:
        FeaturesTuple = list()
        x = self.Conv1(x)
        FeaturesTuple.append(x)
        x = self.Layer1(x)
        FeaturesTuple.append(x)
        x = self.Layer2(x)
        FeaturesTuple.append(x)
        x = self.Layer3(x)
        FeaturesTuple.append(x)
        x = self.Layer4(x)
        FeaturesTuple.append(x)
        return FeaturesTuple
    

@registerClsModels("regnetx_200mf")
def regNetX200MF(**kwargs: Any):
    return RegNet(Bottleneck, [1, 1, 4, 7], [24, 56, 152, 368], group_width=8, **kwargs)


@registerClsModels("regnety_200mf")
def regNetY200MF(**kwargs: Any):
    return RegNet(Bottleneck, [1, 1, 4, 7], [24, 56, 152, 368], group_width=8, SERatio=4, **kwargs)


@registerClsModels("regnetx_400mf")
def regNetX400MF(**kwargs: Any):
    return RegNet(Bottleneck, [1, 2, 7, 12], [32, 64, 160, 384], group_width=16, **kwargs)


@registerClsModels("regnety_400mf")
def regNetY400MF(**kwargs: Any):
    return RegNet(Bottleneck, [1, 3, 6, 6], [48, 104, 208, 440], group_width=8, SERatio=4, **kwargs)


@registerClsModels("regnetx_600mf")
def regNetX600MF(**kwargs: Any):
    return RegNet(Bottleneck, [1, 3, 5, 7], [48, 96, 240, 528], group_width=24, **kwargs)


@registerClsModels("regnety_600mf")
def regNetY600MF(**kwargs: Any):
    return RegNet(Bottleneck, [1, 3, 7, 4], [48, 112, 256, 608], group_width=16, SERatio=4, **kwargs)


@registerClsModels("regnetx_800mf")
def regNetX800MF(**kwargs: Any):
    return RegNet(Bottleneck, [1, 3, 7, 5], [64, 128, 288, 672], group_width=16, **kwargs)


@registerClsModels("regnety_800mf")
def regNetY800MF(**kwargs: Any):
    return RegNet(Bottleneck, [1, 3, 8, 2], [64, 128, 320, 768], group_width=16, SERatio=4, **kwargs)


@registerClsModels("regnetx_1600mf")
def regNetX1600MF(**kwargs: Any):
    return RegNet(Bottleneck, [2, 4, 10, 2], [72, 168, 408, 912], group_width=24, **kwargs)


@registerClsModels("regnety_1600mf")
def regNetY1600MF(**kwargs: Any):
    return RegNet(Bottleneck, [2, 6, 17, 2], [48, 120, 336, 888], group_width=24, SERatio=4, **kwargs)


@registerClsModels("regnetx_3200mf")
def regNetX3200MF(**kwargs: Any):
    return RegNet(Bottleneck, [2, 6, 15, 2], [96, 192, 432, 1008], group_width=48, **kwargs)


@registerClsModels("regnety_3200mf")
def regNetY3200MF(**kwargs: Any):
    return RegNet(Bottleneck, [2, 5, 13, 1], [72, 216, 576, 1512], group_width=24, SERatio=4, **kwargs)


@registerClsModels("regnetx_4000mf")
def regNetX4000MF(**kwargs: Any):
    return RegNet(Bottleneck, [2, 5, 14, 2], [80, 240, 560, 1360], group_width=40, **kwargs)


@registerClsModels("regnety_4000mf")
def regNetY4000MF(**kwargs: Any):
    return RegNet(Bottleneck, [2, 6, 12, 2], [128, 192, 512, 1088], group_width=64, SERatio=4, **kwargs)


@registerClsModels("regnetx_6400mf")
def regNetX6400MF(**kwargs: Any):
    return RegNet(Bottleneck, [2, 4, 10, 1], [168, 392, 784, 1624], group_width=56, **kwargs)


@registerClsModels("regnety_6400mf")
def regNetY6400MF(**kwargs: Any):
    return RegNet(Bottleneck, [2, 7, 14, 2], [144, 288, 576, 1296], group_width=72, SERatio=4, **kwargs)


@registerClsModels("regnetx_8gf")
def regNetX8GF(**kwargs: Any):
    return RegNet(Bottleneck, [2, 5, 15, 1], [80, 240, 720, 1920], group_width=120, **kwargs)


@registerClsModels("regnety_8gf")
def regNetY8GF(**kwargs: Any):
    return RegNet(Bottleneck, [2, 4, 10, 1], [168, 448, 896, 2016], group_width=56, SERatio=4, **kwargs)


@registerClsModels("regnetx_12gf")
def regNetX12GF(**kwargs: Any):
    return RegNet(Bottleneck, [2, 5, 11, 1], [224, 448, 896, 2240], group_width=112, **kwargs)


@registerClsModels("regnety_12gf")
def regNetX12GF(**kwargs: Any):
    return RegNet(Bottleneck, [2, 5, 11, 1], [224, 448, 896, 2240], group_width=112, SERatio=4, **kwargs)


@registerClsModels("regnetx_16gf")
def regNetX16GF(**kwargs: Any):
    return RegNet(Bottleneck, [2, 6, 13, 1], [256, 512, 896, 2048], group_width=128, **kwargs)


@registerClsModels("regnety_16gf")
def regNetX16GF(**kwargs: Any):
    return RegNet(Bottleneck, [2, 4, 11, 1], [224, 448, 1232, 3024], group_width=112, SERatio=4, **kwargs)


@registerClsModels("regnetx_32gf")
def regNetX32GF(**kwargs: Any):
    return RegNet(Bottleneck, [2, 7, 13, 1], [336, 672, 1344, 2520], group_width=168, **kwargs)


@registerClsModels("regnety_32gf")
def regNetX32GF(**kwargs: Any):
    return RegNet(Bottleneck, [2, 5, 12, 1], [232, 696, 1392, 3712], group_width=232, SERatio=4, **kwargs)