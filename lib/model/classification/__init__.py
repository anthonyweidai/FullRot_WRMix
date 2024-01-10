from typing import Any
from .base import BaseEncoder
from ...utils import colorText, importModule


ClsModelRegistry = {}


def registerClsModels(Name):
    def registerModelClass(Cls):
        if Name in ClsModelRegistry:
            raise ValueError("Cannot register duplicate model ({})".format(Name))
        ClsModelRegistry[Name] = Cls
        return Cls
    return registerModelClass


def buildClassificationModel(opt, **kwargs: Any):
    Model: BaseEncoder = None
    if opt.model_name in ClsModelRegistry:
        Model = ClsModelRegistry[opt.model_name](opt=opt, **kwargs)
    else:
        SupportedModels = list(ClsModelRegistry.keys())
        SuppModelStr = "Supported models are:"
        for i, Name in enumerate(SupportedModels):
            SuppModelStr += "\n\t {}: {}".format(i, colorText(Name))
    
    return Model


# Automatically import the models
importModule(__file__, RelativePath="lib.model.classification.")
importModule(__file__, RelativePath="lib.model.classification.classic.", SubFold='/classic/')