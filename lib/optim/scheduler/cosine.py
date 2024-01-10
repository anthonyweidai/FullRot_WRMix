import math
from typing import Optional, Any

from .baseScheduler import BaseLRScheduler
from . import registerScheduler


@registerScheduler("cosine")
class CosineScheduler(BaseLRScheduler):
    """
    Cosine learning rate scheduler: https://arxiv.org/abs/1608.03983
    """
    def __init__(
        self,
        opt,
        MaxEpochs=500,
        MaxLrRate=2e-3,
        MinLrRate=2e-4,
        WarmupLrRate=2e-4,
        WarmupIterations=3000,
        MaxIter=300000,
        IterBased: Optional[bool]=False,
        **kwargs: Any,
        ) -> None:
        super(CosineScheduler, self).__init__(opt)
        self.Milestones = opt.milestones
        self.IterTemp = 0
        
        WarmupIterations = WarmupIterations

        self.MinLrRate = MinLrRate
        self.MaxLrRate = MaxLrRate

        self.WarmupIterations = max(WarmupIterations, 0)
        if self.WarmupIterations > 0:
            self.WarmupLrRate = WarmupLrRate
            self.warmup_step = (self.MaxLrRate - self.WarmupLrRate) / self.WarmupIterations

        self.Period = MaxIter - self.WarmupIterations + 1 if IterBased else MaxEpochs

        self.IterBased = IterBased
        
    def getLr(self, Epoch: int, CurrIter: int) -> float:
        if Epoch == self.Milestones:
            self.IterTemp = CurrIter - 1
            self.WarmupIterations += CurrIter
        
        if CurrIter < self.WarmupIterations:
            Currlr = self.WarmupLrRate + (CurrIter - self.IterTemp) * self.warmup_step
        else:
            if self.IterBased:
                CurrIter = CurrIter - self.WarmupIterations
                Currlr = self.MinLrRate + 0.5 * (self.MaxLrRate - self.MinLrRate) * (1 + math.cos(math.pi * CurrIter / self.Period))
            else:
                Currlr = self.MinLrRate + 0.5 * (self.MaxLrRate - self.MinLrRate) * (1 + math.cos(math.pi * Epoch / self.Period))
        return max(0.0, Currlr)

    def __repr__(self) -> str:
        ReprStr = '{}('.format(self.__class__.__name__)
        ReprStr += '\n \t MinLrRate={}\n \t MaxLrRate={}\n \t Period={}'.format(self.MinLrRate, self.MaxLrRate, self.Period)
        if self.WarmupIterations > 0:
            ReprStr += '\n \t WarmupLrRate={}\n \t warmup_iters={}'.format(self.WarmupLrRate, self.WarmupIterations)

        ReprStr += '\n )'
        return ReprStr

