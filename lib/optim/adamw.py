from typing import Any
from torch.optim import AdamW
from . import registerOptimizer
from .baseOptim import BaseOptim


@registerOptimizer("adamw")
class AdamWOptimizer(BaseOptim, AdamW):
    """
    `AdamW <https://arxiv.org/abs/1711.05101>`_ optimizer
    """

    def __init__(self, opt, NetParam, **kwargs: Any) -> None:
        BaseOptim.__init__(self, opt, **kwargs)
        AdamW.__init__(
            self,
            params=NetParam,
            lr=self.Lr,
            betas=(opt.beta1, opt.beta2),
            eps=self.Eps,
            weight_decay=self.WeightDecay,
            amsgrad=opt.amsgrad,
        )
