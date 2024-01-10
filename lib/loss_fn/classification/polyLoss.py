from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from . import registerClsLossFn
from ..baseCriteria import BaseCriteria


@registerClsLossFn("poly_cross_entropy")
class Poly1CrossEntropyLoss(BaseCriteria):
    def __init__(self, opt, Epsilon: float = 1.0, **kwargs: Any):
        """
        Create instance of Poly1CrossEntropyLoss
        :param Epsilon:
        :param weight: manual rescaling weight for each class, passed to Cross-Entropy loss
        """
        super(Poly1CrossEntropyLoss, self).__init__(opt, **kwargs)
        self.Epsilon = Epsilon

    def forward(self, Input: Tensor, Prediction: Tensor, Target: Tensor) -> Tensor:
        """
        Forward pass
        :param Prediction: tensor of shape [N, NumClasses]
        :param Target: tensor of shape [N]
        :return: poly cross-entropy loss
        """
        # use label smoothing only for training
        LabelSmoothing = self.LabelSmoothing if self.training else 0.0
        
        Weight = self.weightForward(Target)
        CE = F.cross_entropy(
            input=Prediction,
            target=Target,
            weight=Weight,
            reduction='none',
            label_smoothing=LabelSmoothing,
            ignore_index=self.opt.ignore_idx,
            )
        
        LabelsOnehot = F.one_hot(Target, num_classes=self.NumClasses).to(device=Prediction.device,
                                                                           dtype=Prediction.dtype)
        Pt = torch.sum(LabelsOnehot * F.softmax(Prediction, dim=-1), dim=-1)
        
        Loss = CE + self.Epsilon * (1 - Pt)
        
        return self.lossRedManager(Loss)
    
    
@registerClsLossFn("poly_focal")
class Poly1FocalLoss(BaseCriteria):
    def __init__(self,
                 opt,
                 Epsilon: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 pos_weight: Tensor = None,
                 label_is_onehot: bool = False,
                 **kwargs: Any,
                 ):
        """
        Create instance of Poly1FocalLoss
        :param Epsilon: poly loss Epsilon
        :param alpha: focal loss alpha
        :param gamma: focal loss gamma
        :param label_is_onehot: set to True if labels are one-hot encoded
        """
        super(Poly1FocalLoss, self).__init__(opt, **kwargs)
        self.Epsilon = Epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_is_onehot = label_is_onehot

    def forward(self, Input: Tensor, Prediction: Tensor, Target: Tensor) -> Tensor:
        """
        Forward pass
        :param Prediction: output of neural netwrok of shape [N, NumClasses] or [N, NumClasses, ...]
        :param Target: ground truth tensor of shape [N] or [N, ...] with class ids if label_is_onehot was set to False, otherwise 
            one-hot encoded tensor of same shape as Prediction
        :return: poly focal loss
        """
        # focal loss implementation taken from
        # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py

        p = torch.sigmoid(Prediction)

        if not self.label_is_onehot:
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, NumClasses]
            if Target.ndim == 1:
                Target = F.one_hot(Target, num_classes=self.NumClasses)

            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, NumClasses, ...]
            else:
                Target = F.one_hot(Target.unsqueeze(1), self.NumClasses).transpose(1, -1).squeeze_(-1)

        Target = Target.to(device=Prediction.device,
                           dtype=Prediction.dtype)

        Weight = self.weightForward(Target)
        CeLoss = F.binary_cross_entropy_with_logits(input=Prediction,
                                                     target=Target,
                                                     reduction="none",
                                                     weight=Weight,
                                                     pos_weight=self.pos_weight)
        Pt = Target * p + (1 - Target) * (1 - p)
        FL = CeLoss * ((1 - Pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * Target + (1 - self.alpha) * (1 - Target)
            FL = alpha_t * FL

        Loss = FL + self.Epsilon * torch.pow(1 - Pt, self.gamma + 1)

        return self.lossRedManager(Loss)
