from typing import Any

from .classification import buildClassificationLossFn
from .segmentation import buildSegmentationLossFn


SupportedTasks = ["segmentation", "classification"]

def getLossFn(opt, Task=None, **kwargs: Any):
    if not Task:
        Task = opt.task
    
    LossFn = None
    if Task == "classification":
        LossFn = buildClassificationLossFn(opt, Task=Task, **kwargs)
    elif Task == "segmentation":
        LossFn = buildSegmentationLossFn(opt, Task=Task, **kwargs) # Task value for mixupMask
    else:
        TaskStr = 'Got {} as a task. Unfortunately, we do not support it yet.' \
                '\nSupported tasks are:'.format(opt.task)
        for i, Name in enumerate(SupportedTasks):
            TaskStr += "\n\t {}: {}".format(i, Name)
    return LossFn