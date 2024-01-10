from typing import Any
from ..utils import colorText, importModule


SupMethodRegistry = {}

SEPARATOR = ":"


def registerDataset(Name, Task):
    def registerMethodClass(Cls):
        if Name in SupMethodRegistry:
            raise ValueError("Cannot register duplicate model ({})".format(Name))
        SupMethodRegistry[Name + SEPARATOR + Task] = Cls
        return Cls
    return registerMethodClass


def getMyDataset(opt, ImgPaths, TargetSet=None, **kwargs: Any):
    DatasetMethod = None
    RegName = opt.sup_method + SEPARATOR + opt.task
    if RegName in SupMethodRegistry:
        DatasetMethod = SupMethodRegistry[RegName](opt, ImgPaths, TargetSet=TargetSet, **kwargs)
    else:
        Supported = list(SupMethodRegistry.keys())
        SuppStr = "Supported datasets are:"
        for i, Name in enumerate(Supported):
            SuppStr += "\n\t {}: {}".format(i, colorText(Name))
    return DatasetMethod


# Automatically import the dataset classes
importModule(__file__, RelativePath="lib.data.")
importModule(__file__, RelativePath="lib.data.classification.", SubFold='/classification/')
importModule(__file__, RelativePath="lib.data.segmentation.", SubFold='/segmentation/')