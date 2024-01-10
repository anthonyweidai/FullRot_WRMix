from typing import Any

from ...utils import colorText, importModule


LossFnRegisty = {}


def registerClsLossFn(Name):
    def registerLossFnClass(Fn):
        if Name in LossFnRegisty:
            raise ValueError("Cannot register duplicate classification loss function ({})".format(Name))
        LossFnRegisty[Name] = Fn
        return Fn
    return registerLossFnClass


def buildClassificationLossFn(opt, **kwargs: Any):
    LossFn = None
    if opt.cls_loss_name in LossFnRegisty:
        LossFn = LossFnRegisty[opt.cls_loss_name](opt=opt, **kwargs)
    else:
        TempList = list(LossFnRegisty.keys())
        TempStr = "Supported loss functions are:"
        for i, Name in enumerate(TempList):
            TempStr += "\n\t {}: {}".format(i, colorText(Name))
    return LossFn


# automatically import different loss functions
importModule(__file__, RelativePath="lib.loss_fn.classification.")