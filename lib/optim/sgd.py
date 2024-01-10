from typing import Any
from torch.optim import SGD
from . import registerOptimizer
from .baseOptim import BaseOptim


@registerOptimizer("sgd")
class SGDOptimizer(BaseOptim, SGD):
    """
    `SGD <http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf>`_ optimizer
    """
    def __init__(self, opt, NetParam, **kwargs: Any) -> None:
        BaseOptim.__init__(self, opt, **kwargs)
        momentum = opt.momentum

        SGD.__init__(
            self,
            params=NetParam,
            lr=self.Lr,
            momentum=momentum,
            weight_decay=self.WeightDecay,
            nesterov=opt.nesterov,
        )
        