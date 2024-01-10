from torch import Tensor
from typing import Union, Tuple, Dict

from . import registerSegModels
from .baseSeg import BaseSegmentation
from .heads import buildSegmentationHead


@registerSegModels("encoder_decoder")
class SegEncoderDecoder(BaseSegmentation):
    """
    This class defines a encoder-decoder architecture for the task of semantic segmentation. 
    Different segmentation heads (e.g., PSPNet and DeepLabv3) can be used

    Args:
        opt: command-line arguments
        Encoder (BaseEncoder): Backbone network (e.g., ResNext)
    """

    def __init__(self, opt, Encoder, **kwargs) -> None:
        super().__init__(opt, Encoder)
        # delete layers that are not required in segmentation network
        self.Encoder.Classifier = None

        self.SegHead = buildSegmentationHead(opt, ModelConfigDict=self.Encoder.ModelConfigDict)

    def forward(self, x: Tensor, **kwargs) -> Union[Tuple[Tensor, Tensor], Tensor]:
        if isinstance(x, Dict):
            Input = x["image"]
        elif isinstance(x, Tensor):
            Input = x
        else:
            raise NotImplementedError(
                "Input to segmentation should be either a Tensor or a Dict of Tensors"
            )
        
        FeaturesTuple = self.Encoder.forwardTuple(Input)
        return self.SegHead(FeaturesTuple, **kwargs)