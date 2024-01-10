from torch import nn, Tensor
from typing import Optional, Tuple

from ..classification.base import BaseEncoder
from ..layers import NormLayerTuple


class BaseSegmentation(nn.Module):
    """Base class for segmentation networks"""

    def __init__(self, opt, Encoder: Optional[BaseEncoder]) -> None:
        super(BaseSegmentation, self).__init__()
        assert isinstance(
            Encoder, BaseEncoder
        ), "Encoder should be an instance of BaseEncoder"
        self.opt = opt
        self.Encoder: BaseEncoder = Encoder

    def profileModel(self, Input: Tensor) -> Optional[Tuple[Tensor, float, float]]:
        """
        Child classes must implement this function to compute FLOPs and parameters
        """
        raise NotImplementedError

    def freezeNormLayers(self) -> None:
        for m in self.modules():
            if isinstance(m, NormLayerTuple):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.training = False
