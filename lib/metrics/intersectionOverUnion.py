import torch
from torch import Tensor
from typing import Optional, Tuple, Union, Dict

from .. utils import tensor2PythonFloat


def computeIoUBatch(
    Prediction: Union[Tuple[Tensor, Tensor], Tensor],
    Target: Tensor,
    Epsilon: Optional[float] = 1e-7,
):
    if isinstance(Prediction, Dict):
        Prediction = Prediction['mask']
        Target = Target['mask']
    
    if isinstance(Prediction, Tuple) and len(Prediction) == 2:
        Mask = Prediction[0]
        assert isinstance(Mask, Tensor)
    elif isinstance(Prediction, Tensor):
        Mask = Prediction
        assert isinstance(Mask, Tensor)
    else:
        raise NotImplementedError(
            "For computing loss for segmentation task, we need Prediction to be an instance of Tuple or Tensor"
        )

    NumClasses = Mask.shape[1]
    # PredMask = torch.max(Mask, dim=1)[1]
    PredMask = torch.argmax(Mask, dim=1)
    if Target.ndim == 4:
        Target = torch.argmax(Target, dim=1)
    assert (
        PredMask.dim() == 3
    ), "Predicted Mask tensor should be 3-dimensional (B x H x W)"

    PredMask = PredMask.byte()
    Target = Target.byte()

    # shift by 1 so that 255 is 0
    PredMask += 1
    Target += 1

    PredMask = PredMask * (Target > 0)
    Inter = PredMask * (PredMask == Target)
    AreaInter = torch.histc(Inter.float(), bins=NumClasses, min=1, max=NumClasses)
    AreaPred = torch.histc(PredMask.float(), bins=NumClasses, min=1, max=NumClasses)
    AreaMask = torch.histc(Target.float(), bins=NumClasses, min=1, max=NumClasses)
    AreaUnion = AreaPred + AreaMask - AreaInter + Epsilon

    AreaInter = tensor2PythonFloat(AreaInter, IsDistributed=False)
    AreaUnion = tensor2PythonFloat(AreaUnion, IsDistributed=False)
    
    return AreaInter, AreaUnion
