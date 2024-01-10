from torch import nn, Tensor
from timm.models.layers import drop_path


class StochasticDepth(nn.Module):
    def __init__(self, DProb: float) -> None:
        super().__init__()
        self.DProb = DProb

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.DProb, self.training)
    
    def profileModule(self, Input: Tensor):
        _, in_channels, in_h, in_w = Input.size()
        MACs = in_channels * in_h * in_w # one multiplication for each element
        return Input, 0.0, MACs
    
    
class Dropout(nn.Dropout):
    def __init__(self, p: float=0.5, inplace: bool=False):
        super(Dropout, self).__init__(p=p, inplace=inplace)

    def profileModule(self, Input: Tensor):
        Input = self.forward(Input)
        return Input, 0.0, 0.0


class Dropout2d(nn.Dropout2d):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__(p=p, inplace=inplace)

    def profile_module(self, input: Tensor):
        return input, 0.0, 0.0
