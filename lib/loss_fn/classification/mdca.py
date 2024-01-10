import torch
from torch import nn, Tensor

from .focal import FocalLossV2
from . import registerClsLossFn
from ..baseCriteria import BaseCriteria

    
# Paper: A Stitch in Time Saves Nine:A Train-Time Regularizing Loss 
# for Improved Neural Network Calibration
class MDCA(torch.nn.Module):
    def __init__(self, Device):
        super(MDCA, self).__init__()
        self.Device = Device

    def forward(self, Prediction, Target):
        Prediction = torch.softmax(Prediction, dim=1)
        # [batch, classes]
        Loss = torch.tensor(0.0).to(self.Device)
        _, classes = Prediction.shape
        for c in range(classes):
            avg_count = (Target == c).float().mean()
            avg_conf = torch.mean(Prediction[:,c])
            Loss += torch.abs(avg_conf - avg_count)
        denom = classes
        Loss /= denom
        return Loss


@registerClsLossFn("mdca_focal")
@registerClsLossFn("mdca_cross_entropy")
class ClassficationAndMDCA(BaseCriteria):
    def __init__(self, opt, alpha=0.1, gamma=1.0, beta=1.0, **kwargs):
        super(ClassficationAndMDCA, self).__init__(opt, **kwargs)
        self.beta = beta
        
        if "focal" in opt.cls_loss_name:
            self.ClsLoss = FocalLossV2(opt, alpha=alpha, gamma=gamma)
        else:
            self.ClsLoss = nn.CrossEntropyLoss(reduction=self.Reduction, 
                                               label_smoothing=self.LabelSmoothing)
            
        self.MDCA = MDCA(opt.device)

    def forward(self, Input: Tensor, Prediction: Tensor, Target: Tensor):
        ClsLoss = self.ClsLoss(Prediction, Target)
        CalLoss = self.MDCA(Prediction, Target)
        return ClsLoss + self.beta * CalLoss