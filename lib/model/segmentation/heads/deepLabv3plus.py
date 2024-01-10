from typing import Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from . import registerSegmentationHead
from .modules import ASPP
from .baseSegHead import BaseSegHead
from ...layers import BaseConv2d, initWeight
from ....utils import pair


@registerSegmentationHead("deeplabv3plus")
class DeeplabV3Plus(BaseSegHead):
    """
    This class defines the segmentation head in 
    `DeepLabv3 architecture <https://arxiv.org/abs/1706.05587>`_
    """
    def __init__(self, opt, **kwargs: Any) -> None:
        AtrousRates = (12, 24, 36) # (6, 12, 18)
        OutChannels = 512
        IsSepConv = opt.use_sep_conv
        DropRate = 0.1

        super().__init__(opt, **kwargs)
        # Decoder
        
        self.ReduceIdx = self.getLastIdxFromStage(self.ModelConfigDict, -2)
        ReduceInChannels = self.getChannelsbyStage(self.ModelConfigDict, self.ReduceIdx)
        self.Reduce = BaseConv2d(ReduceInChannels, 48, 1, 1, BNorm=True, ActLayer=nn.ReLU)
    
        self.Aspp = nn.Sequential()
        self.Aspp.add_module(
            name="aspp_layer",
            module=ASPP(
                self.ModelConfigDict[list(self.ModelConfigDict)[-1]]['out'],
                OutChannels,
                AtrousRates,
                IsSepConv,
                DropRate,
            ),
        )
        
        InChannels = 304 if OutChannels == 256 else 560
        self.Classifier = nn.Sequential(
            BaseConv2d(InChannels, OutChannels, 3, 1, BNorm=True, ActLayer=nn.ReLU),
            BaseConv2d(OutChannels, OutChannels, 3, 1, BNorm=True, ActLayer=nn.ReLU),
            BaseConv2d(OutChannels, self.NumClasses, 1),
        )

        if opt.init_weight:
            self.apply(initWeight)

    def forwardSegHead(self, FeaturesTuple: list) -> Tensor:
        x_ = self.Reduce(FeaturesTuple[self.ReduceIdx])
        x = self.Aspp(FeaturesTuple[-1])
        x = F.interpolate(x, size=x_.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat((x, x_), dim=1)
        return self.Classifier(x)
    