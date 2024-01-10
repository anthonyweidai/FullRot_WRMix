from typing import Any
from .classification import buildClassificationModel
from .segmentation import buildSegmentationModel


SUPPORTED_TASKS = ["classification", "segmentation"]

def getModel(opt, **kwargs: Any):
    Model = None
    if opt.task == "classification":
        Model = buildClassificationModel(opt, **kwargs)
    elif opt.task == "segmentation":
        Model = buildSegmentationModel(opt, **kwargs)
    else:
        TaskStr = 'Got {} as a task. Unfortunately, we do not support it yet.' \
                   '\nSupported tasks are:'.format(opt.task)
        for i, Name in enumerate(SUPPORTED_TASKS):
            TaskStr += "\n\t {}: {}".format(i, Name)
    return Model