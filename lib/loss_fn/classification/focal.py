from typing import Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.cuda.amp as amp

from . import registerClsLossFn
from ..baseCriteria import BaseCriteria

        
##
# version 1: use torch.autograd
class FocalLossV1(BaseCriteria):
    def __init__(self, opt, alpha=0.25, gamma=2, **kwargs: Any):
        super(FocalLossV1, self).__init__(opt, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, Input: Tensor, Prediction: Tensor, Target: Tensor) -> Tensor:
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV1()
            >>> Prediction = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> Loss = criteria(Prediction, lbs)
        '''
        Target = F.one_hot(Target, num_classes=Prediction.shape[-1])
        probs = torch.sigmoid(Prediction)
        coeff = torch.abs(Target - probs).pow(self.gamma).neg()
        log_probs = torch.where(Prediction >= 0,
                F.softplus(Prediction, -1, 50),
                Prediction - F.softplus(Prediction, 1, 50))
        log_1_probs = torch.where(Prediction >= 0,
                -Prediction + F.softplus(Prediction, -1, 50),
                -F.softplus(Prediction, 1, 50))
        Loss = Target * self.alpha * log_probs + (1. - Target) * (1. - self.alpha) * log_1_probs
        Loss = Loss * coeff

        return self.lossRedManager(Loss)


##
# version 2: user derived grad computation
class FocalSigmoidLossFuncV2(torch.autograd.Function):
    '''
    compute backward directly for better numeric stability
    '''
    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, Prediction, Target, alpha, gamma):
        #  Prediction = Prediction.float()

        probs = torch.sigmoid(Prediction)
        coeff = (Target - probs).abs_().pow_(gamma).neg_()
        log_probs = torch.where(Prediction >= 0,
                F.softplus(Prediction, -1, 50),
                Prediction - F.softplus(Prediction, 1, 50))
        log_1_probs = torch.where(Prediction >= 0,
                -Prediction + F.softplus(Prediction, -1, 50),
                -F.softplus(Prediction, 1, 50))
        ce_term1 = log_probs.mul_(Target).mul_(alpha)
        ce_term2 = log_1_probs.mul_(1. - Target).mul_(1. - alpha)
        ce = ce_term1.add_(ce_term2)
        loss = ce * coeff

        ctx.vars = (coeff, probs, ce, Target, gamma, alpha)

        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        '''
        compute gradient of focal loss
        '''
        (coeff, probs, ce, Target, gamma, alpha) = ctx.vars

        d_coeff = (Target - probs).abs_().pow_(gamma - 1.).mul_(gamma)
        d_coeff.mul_(probs).mul_(1. - probs)
        d_coeff = torch.where(Target < probs, d_coeff.neg(), d_coeff)
        term1 = d_coeff.mul_(ce)

        d_ce = Target * alpha
        d_ce.sub_(probs.mul_((Target * alpha).mul_(2).add_(1).sub_(Target).sub_(alpha)))
        term2 = d_ce.mul(coeff)

        grads = term1.add_(term2)
        grads.mul_(grad_output)

        return grads, None, None, None


@registerClsLossFn("focal")
class FocalLossV2(BaseCriteria):
    def __init__(self, opt, alpha=0.25, gamma=2, **kwargs: Any):
        super(FocalLossV2, self).__init__(opt, **kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, Input: Tensor, Prediction: Tensor, Target: Tensor) -> Tensor:
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV2()
            >>> Prediction = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> Loss = criteria(Prediction, lbs)
        '''
        Target = F.one_hot(Target, num_classes=Prediction.shape[-1])
        Loss = FocalSigmoidLossFuncV2.apply(Prediction, Target, self.alpha, self.gamma)
        
        return self.lossRedManager(Loss)
    