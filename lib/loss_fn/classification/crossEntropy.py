from typing import Any

from torch import Tensor
from torch.nn import functional as F

from . import registerClsLossFn
from ..baseCriteria import BaseCriteria


@registerClsLossFn("cross_entropy")
class ClsCrossEntropy(BaseCriteria):
    def __init__(self, opt, **kwargs: Any):
        super(ClsCrossEntropy, self).__init__(opt, **kwargs)

    def forward(self, Input: Tensor, Prediction: Tensor, Target: Tensor) -> Tensor:
        Weight = self.weightForward(Target)
        
        # use label smoothing only for training, # for validation, compute standard CE loss
        LabelSmoothing = self.LabelSmoothing if self.training else 0.0

        return F.cross_entropy(
            input=Prediction,
            target=Target,
            weight=Weight,
            ignore_index=self.opt.ignore_idx,
            reduction=self.Reduction,
            label_smoothing=LabelSmoothing
        )

    def __repr__(self):
        return "{}(\n\IgnoreIdx={}\n\tclass_wts={}\n\LabelSmoothing={}\n)".format(
            self.__class__.__name__,
            self.IgnoreIdx,
            self.UseClsWts,
            self.LabelSmoothing,
        )