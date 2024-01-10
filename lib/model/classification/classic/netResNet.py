from typing import Type, Any, Union, List, Optional

import torch.nn as nn
from torch import Tensor

from .. import registerClsModels
from ..base import BaseEncoder
from ...misc import moduleProfile
from ...layers import BaseConv2d, Linearlayer, initWeight
from ....utils import setMethod


class BasicBlock(nn.Module):
    Expansion: int = 1

    def __init__(
        self,
        InPlanes: int,
        Planes: int,
        Stride: int = 1,
        Downsample: Optional[nn.Module] = None,
        Groups: int = 1,
        BaseWidth: int = 64,
        Dilation: int = 1,
    ) -> None:
        super().__init__()
        if Groups != 1 or BaseWidth != 64:
            raise ValueError('BasicBlock only supports groups=1 and BaseWidth=64')
        if Dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.Downsample layers Downsample the input when stride != 1
        self.Conv1 = BaseConv2d(InPlanes, Planes, 3, Stride, BNorm=True, ActLayer=nn.ReLU)
        self.Conv2 = BaseConv2d(Planes, Planes, 1, BNorm=True)
        
        self.Relu = nn.ReLU(inplace=True)
        self.Downsample = Downsample
        self.Stride = Stride

    def forward(self, x: Tensor) -> Tensor:
        Identity = x

        Out = self.Conv1(x)

        Out = self.Conv2(Out)

        if self.Downsample is not None:
            Identity = self.Downsample(x)

        Out += Identity # Skip connection in each Block
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
        
        if self.Downsample is not None:
            Output, ParamsTemp, MACsTemp = moduleProfile(module=self.Downsample, x=Input)
            Params += ParamsTemp
            MACs += MACsTemp
        
        return Output, Params, MACs


class Bottleneck(nn.Module):
    Expansion: int = 4

    def __init__(
        self,
        InPlanes: int,
        Planes: int,
        Stride: int = 1,
        Downsample: Optional[nn.Module] = None,
        Groups: int = 1,
        BaseWidth: int = 64,
        Dilation: int = 1,
    ) -> None:
        super().__init__()
        width = int(Planes * (BaseWidth / 64.)) * Groups
        # Both self.conv2 and self.Downsample layers Downsample the input when stride != 1
        self.Conv1 = BaseConv2d(InPlanes, width, 1, BNorm=True, ActLayer=nn.ReLU)
        self.Conv2 = BaseConv2d(width, width, 3, Stride, groups=Groups, dilation=Dilation, BNorm=True, ActLayer=nn.ReLU)
        self.Conv3 = BaseConv2d(width, Planes * self.Expansion, 1, BNorm=True)
        self.Relu = nn.ReLU(inplace=True)
        self.Downsample = Downsample # projection, not identity mapping
        self.Stride = Stride

    def forward(self, x: Tensor) -> Tensor:
        Identity = x

        Out = self.Conv1(x)
        Out = self.Conv2(Out)
        Out = self.Conv3(Out)

        if self.Downsample is not None:
            Identity = self.Downsample(x)

        Out += Identity # Skip connection in each Block
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
        
        Output, ParamsTemp, MACsTemp = moduleProfile(module=self.Conv3, x=Output)
        Params += ParamsTemp
        MACs += MACsTemp
        
        if self.Downsample is not None:
            Output, ParamsTemp, MACsTemp = moduleProfile(module=self.Downsample, x=Input)
            Params += ParamsTemp
            MACs += MACsTemp
        
        return Output, Params, MACs


class ResNet(BaseEncoder):
    def __init__(
        self,
        opt,
        Block: Type[Union[BasicBlock, Bottleneck]],
        Layers: List[int],
        Groups: int = 1,
        WidthPerGroup: int = 64,
        ReplaceStrideWithFilation: Optional[List[bool]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(opt, **kwargs)
        
        self.InPlanes = 64
        self.Dilation = 1
        if ReplaceStrideWithFilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            ReplaceStrideWithFilation = [False, False, False]
        if len(ReplaceStrideWithFilation) != 3:
            raise ValueError(
                "ReplaceStrideWithFilation should be None "
                f"or a 3-element tuple, got {ReplaceStrideWithFilation}"
            )        
        self.Groups = Groups
        self.BaseWidth = WidthPerGroup
        
        InChannels = 3
        OutChannels = self.InPlanes
        self.Conv1 = nn.Sequential(
            BaseConv2d(InChannels, OutChannels, kernel_size=7, stride=2, padding=3, BNorm=True, ActLayer=nn.ReLU),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.ModelConfigDict['conv1'] = {'in': InChannels, 'out': OutChannels, 'stage': 1} # with stage 1 and 2

        StageCount = 2
        StrideList = [1, 2, 2, 2]
        OutChannelsList = [64, 128, 256, 512]
        for i in range(4):
            if StrideList[i] >= 2:
                StageCount += 1
            
            InChannels = OutChannels
            OutChannels = OutChannelsList[i]
            Layer, OutChannels = self.makeLayer(Block, OutChannels, Layers[i], StrideList[i], 
                                   ReplaceStrideWithFilation[i - 1] if i > 0 else False)
            setMethod(self, 'Layer%d' % (i + 1), Layer)
            self.ModelConfigDict['layer%d' % (i + 1)] = {'in': InChannels, 'out': OutChannels, 'stage': StageCount}

        self.Classifier = Linearlayer(OutChannels, self.NumClasses)

        if opt.init_weight:
            self.apply(initWeight)

    def makeLayer(
        self, 
        Block: Type[Union[BasicBlock, Bottleneck]], 
        Planes: int, 
        Blocks: int,
        Stride: int = 1, 
        Dilate: bool = False,
        ) -> nn.Sequential:
        Downsample = None
        PreviousDilation = self.Dilation
        if Dilate:
            self.Dilation *= Stride
            Stride = 1
        if Stride != 1 or self.InPlanes != Planes * Block.Expansion:
            Downsample = BaseConv2d(self.InPlanes, Planes * Block.Expansion, 1, Stride, BNorm=True)

        Layers = [] # init layers
        Layers.append(Block(self.InPlanes, Planes, Stride, Downsample, self.Groups,
                            self.BaseWidth, PreviousDilation))
        self.InPlanes = Planes * Block.Expansion
        for _ in range(1, Blocks):
            Layers.append(Block(self.InPlanes, Planes, Groups=self.Groups,
                                BaseWidth=self.BaseWidth, Dilation=self.Dilation
                            ))

        return nn.Sequential(*Layers), self.InPlanes

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
    

def resnet(
    Block: Type[Union[BasicBlock, Bottleneck]],
    Layers: List[int],
    opt,
    **kwargs: Any
) -> ResNet:
    model = ResNet(opt, Block, Layers, **kwargs)
    return model


@registerClsModels("1resnet18")
def resNet18(**kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    return resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


@registerClsModels("1resnet34")
def resNet34(**kwargs: Any) -> ResNet:
    return resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


@registerClsModels("1resnet50")
def resNet50(**kwargs: Any) -> ResNet:
    return resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


@registerClsModels("1resnet101")
def resNet101(**kwargs: Any) -> ResNet:
    return resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


@registerClsModels("1resnet152")
def resNet152(**kwargs: Any) -> ResNet:
    return resnet(Bottleneck, [3, 8, 36, 3], **kwargs)


@registerClsModels("1resnext50_32x4d")
def resnext50_32x4d(**kwargs: Any) -> ResNet:
    """ResNeXt-50 32x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.
    """
    return resnet(Bottleneck, [3, 4, 6, 3], Groups=32, WidthPerGroup=4, **kwargs)


@registerClsModels("1resnext101_32x8d")
def resnext101_32x8d(**kwargs: Any) -> ResNet:
    return resnet(Bottleneck, [3, 4, 23, 3], Groups=32, WidthPerGroup=8, **kwargs)


@registerClsModels("1resnext101_64x4d")
def resnext101_64x4d(**kwargs: Any) -> ResNet:
    return resnet(Bottleneck, [3, 4, 23, 3], Groups=64, WidthPerGroup=4, **kwargs)