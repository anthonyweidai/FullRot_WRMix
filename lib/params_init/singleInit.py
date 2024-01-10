from ..utils import colorText


ParamsInitRegistry = {}


def registerParams(Name):
    def registerParamsClass(Cls):
        if Name in ParamsInitRegistry:
            raise ValueError("Cannot register duplicate model ({})".format(Name))
        ParamsInitRegistry[Name] = Cls
        return Cls
    return registerParamsClass


def singleInit(opt):
    SupMethod = None
    if opt.task in ParamsInitRegistry:
        SupMethod = ParamsInitRegistry[opt.task](opt)
    else:
        SupportedModels = list(ParamsInitRegistry.keys())
        SuppModelStr = "Supported models are:"
        for i, Name in enumerate(SupportedModels):
            SuppModelStr += "\n\t {}: {}".format(i, colorText(Name))
    return SupMethod


from .classification import classficationParamsInit
from .segmentation import segmentationParamsInit