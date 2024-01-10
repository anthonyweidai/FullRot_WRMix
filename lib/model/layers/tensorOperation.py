import math
import torch
from torch import Tensor


def shuffleTensor(Feature: Tensor, Mode: int=1) -> Tensor:
    B, C, H, W = Feature.shape
    if Mode == 1:
        Feature = Feature.flatten(2)
        Feature = Feature[:, :, torch.randperm(Feature.shape[-1], device=Feature.device)]
        Feature = Feature.reshape(B, C, H, W)
    else:
        Feature = Feature[:, :, torch.randperm(H, device=Feature.device)]
        Feature = Feature[:, :, :, torch.randperm(W, device=Feature.device)]
    return Feature


def tensorRepeatLike(Feature: Tensor, RefTensor: Tensor, Res: bool=False, Reduct: bool=False) -> Tensor:
        _, _, H, W = RefTensor.shape
        # if Reduct:
        #     Feature = Feature.flatten(2).unsqueeze(3)
        #     _, _, Region, _ = Feature.shape
        #     Step = math.ceil(H / Region)
        #     Feature = Feature.repeat(1, 1, Step, 1)
        #     Feature = Feature[:, :, 0:H, :]
        #     Output = Feature
        # else:
        _, _, HRegion, WRegion = Feature.shape
        HStep = math.ceil(H / HRegion)
        WStep = math.ceil(W / WRegion)
        Feature = Feature.repeat(1, 1, HStep, WStep)
        Feature = Feature[:, :, 0:H, 0:W]
        Output = RefTensor + Feature if Res else Feature
        return Output

