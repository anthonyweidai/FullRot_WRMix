from torch import nn, Tensor
from typing import Any, Tuple

from ...layers import (BaseConv2d, Dropout2d, UpSample, checkExp,
                       getLastIdxFromStage, getChannelsbyStage, callDictbyStage)
from ....utils import pair


class BaseSegHead(nn.Module):
    def __init__(self, opt, ModelConfigDict, **kwargs: Any):
        super().__init__()

        self.opt = opt
        self.NumClasses = opt.seg_num_classes
        self.classifier_dropout = 0.1
        self.OutputStride = 16
        self.ModelConfigDict = ModelConfigDict
        
        self.AuxHead = None
        if opt.use_aux_head:
            self.StageIdx = self.getLastIdxFromStage(ModelConfigDict, -2)
            EncLast2Channels = ModelConfigDict[list(ModelConfigDict)[self.StageIdx]]['out']
            InnerChannels = max(int(EncLast2Channels // 4), 128)
            self.AuxHead = nn.Sequential(
                BaseConv2d(EncLast2Channels, InnerChannels, 3, 1, BNorm=True, ActLayer=nn.ReLU),
                Dropout2d(0.1),
                BaseConv2d(InnerChannels, self.NumClasses, 1, 1, ActLayer=None)
                )

        self.UpsampleSegOut = None
        if self.OutputStride != 1.0:
            self.UpsampleSegOut = UpSample(
                scale_factor=self.OutputStride, mode="bilinear", align_corners=True
            )
        
        self.checkExp = checkExp
        self.getLastIdxFromStage = getLastIdxFromStage
        self.getChannelsbyStage = getChannelsbyStage
        self.callDictbyStage = callDictbyStage
        
    def forwardAuxHead(self, FeaturesTuple) -> Tensor:
        AuxOut = self.AuxHead(FeaturesTuple[self.StageIdx])
        return AuxOut

    def forwardSegHead(self, FeaturesTuple) -> Tensor:
        raise NotImplementedError

    def forward(self, FeaturesTuple, **kwargs) -> Tensor or Tuple[Tensor]:
        Out = self.forwardSegHead(FeaturesTuple)

        if self.UpsampleSegOut is not None:
            # resize the mask based on given size
            MaskSize = pair(self.opt.resize_res)
            if MaskSize is not None:
                self.UpsampleSegOut.scale_factor = None
                self.UpsampleSegOut.size = MaskSize

            Out = self.UpsampleSegOut(Out)

        if self.AuxHead is not None and self.training:
            AuxOut = self.forwardAuxHead(FeaturesTuple)
            return Out, AuxOut
        return Out

    def profileModule(self, x: Tensor) -> Tuple[Tensor, float, float]:
        """
        Child classes must implement this function to compute FLOPs and parameters
        """
        raise NotImplementedError
