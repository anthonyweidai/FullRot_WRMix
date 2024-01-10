from typing import Any

from ..classification import buildClassificationModel
from ...utils import colorText, importModule


SegModelRegistry = {}


def registerSegModels(Name):
    def registerModelClass(Cls):
        if Name in SegModelRegistry:
            raise ValueError("Cannot register duplicate model ({})".format(Name))

        SegModelRegistry[Name] = Cls
        return Cls
    return registerModelClass


def buildSegmentationModel(opt, **kwargs: Any):
    Model = None
    if opt.seg_model_name in SegModelRegistry:
        if 'encoder_decoder' in opt.seg_model_name:
            Encoder = buildClassificationModel(opt)
            Model = SegModelRegistry[opt.seg_model_name](opt=opt, Encoder=Encoder, **kwargs)
        else:
            Model = SegModelRegistry[opt.seg_model_name](opt=opt, **kwargs)
    else:
        SupportedModels = list(SegModelRegistry.keys())
        SuppModelStr = "Supported models are:"
        for i, Name in enumerate(SupportedModels):
            SuppModelStr += "\n\t {}: {}".format(i, colorText(Name))
    return Model


# Automatically import the models
importModule(__file__, RelativePath="lib.model.segmentation.")