import os
from glob import glob

import copy
import math
import random
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from sklearn.model_selection import KFold
from typing import Sequence, Union, Optional, Any, Dict, List

import torch
from torch import Tensor
from torchvision import transforms as T
import torchvision.transforms.functional as F

from ..utils import pair, colorText


def getImgPath(DatasetPath, NumSplit, Mode=1, Shuffle=True):
    # Put images into train set or test set
    if Mode == 1:
        '''
        root/split1/dog_1.png
        root/split1/dog_2.png
        root/split2/cat_1.png
        root/split2/cat_2.png
        '''
        TrainSet, TestSet = [], []
        for i in range(1, NumSplit + 1):
            TestSet.append(glob(DatasetPath + '/' + 'split{}'.format(i) + '/*'))
            
            TrainImgs = []
            for j in range(1, NumSplit + 1):
                if j != i:
                    TrainImgs.extend(glob(DatasetPath + '/' + 'split{}'.format(j) + '/*'))
            TrainSet.append(TrainImgs)
                
    elif Mode == 2:
        '''
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png
        '''
        TrainSet, TestSet = [[] for _ in range(NumSplit)], [[] for _ in range(NumSplit)]
        ClassNames = os.listdir(DatasetPath)
        Kf = KFold(n_splits=NumSplit, shuffle=Shuffle)
        
        for ClassName in ClassNames:
            ImagePath = glob(DatasetPath + '/' + ClassName + '/*')
            IndexList = range(0, len(ImagePath))

            Kf.get_n_splits(IndexList)
            
            for idx, (TrainIndexes, TestIdexes) in enumerate(Kf.split(IndexList)):
                [TrainSet[idx].append(ImagePath[i]) for i in TrainIndexes]
                [TestSet[idx].append(ImagePath[j]) for j in TestIdexes]
                
    elif Mode == 3:
        '''
        root/train/dog/xxx.png
        root/train/dog/xxy.png
        root/train/dog/[...]/xxz.png
        root/val/dog/xxx.png
        root/val/dog/xxy.png
        root/val/dog/[...]/xxz.png
        or,
        root/train/dog/xxx.png
        root/train/dog/xxy.png
        root/train/dog/[...]/xxz.png
        root/test/dog/xxx.png
        root/test/dog/xxy.png
        root/test/dog/[...]/xxz.png
        '''
        TrainSet = glob('%s/%s/*/*' % (DatasetPath, 'train'))
        
        TestSet = []
        ValidStr = ['val', 'test']
        for Name in ValidStr:
            SetPath = '%s/%s/' % (DatasetPath, Name)
            if os.path.isdir(SetPath):
                TestSet = glob(SetPath + '*/*')
                break
            
    elif Mode == 4:
        '''
        root/train/xxx.png
        root/train/xxy.png
        root/train/[...]/xxz.png
        root/val/xxx.png
        root/val/xxy.png
        root/val/[...]/xxz.png
        or,
        root/train/xxx.png
        root/train/xxy.png
        root/train/[...]/xxz.png
        root/test/xxx.png
        root/test/xxy.png
        root/test/[...]/xxz.png
        '''
        TrainSet = glob('%s/%s/*' % (DatasetPath, 'train'))
        
        TestSet = []
        ValidStr = ['val', 'test']
        for Name in ValidStr:
            SetPath = '%s/%s/' % (DatasetPath, Name)
            if os.path.isdir(SetPath):
                TestSet = glob(SetPath + '*')
                break
    return TrainSet, TestSet


def cropFn(Data: Dict, top: int, left: int, height: int, width: int) -> Dict:
    """Helper function for cropping"""
    Img = Data["image"]
    Data["image"] = F.crop(Img, top=top, left=left, height=height, width=width)

    if "mask" in Data:
        Mask = Data.pop("mask")
        Data["mask"] = F.crop(Mask, top=top, left=left, height=height, width=width)

    if "box_coordinates" in Data:
        boxes = Data.pop("box_coordinates")

        area_before_cropping = (boxes[..., 2] - boxes[..., 0]) * (
            boxes[..., 3] - boxes[..., 1]
        )

        boxes[..., 0::2] = np.clip(boxes[..., 0::2] - left, a_min=0, a_max=left + width)
        boxes[..., 1::2] = np.clip(boxes[..., 1::2] - top, a_min=0, a_max=top + height)

        area_after_cropping = (boxes[..., 2] - boxes[..., 0]) * (
            boxes[..., 3] - boxes[..., 1]
        )
        area_ratio = area_after_cropping / (area_before_cropping + 1)

        # keep the boxes whose area is atleast 20% of the area before cropping
        keep = area_ratio >= 0.2

        box_labels = Data.pop("box_labels")

        Data["box_coordinates"] = boxes[keep]
        Data["box_labels"] = box_labels[keep]

    if "instance_mask" in Data:
        assert "instance_coords" in Data

        instance_masks = Data.pop("instance_mask")
        Data["instance_mask"] = F.crop(
            instance_masks, top=top, left=left, height=height, width=width
        )

        instance_coords = Data.pop("instance_coords")
        instance_coords[..., 0::2] = np.clip(
            instance_coords[..., 0::2] - left, a_min=0, a_max=left + width
        )
        instance_coords[..., 1::2] = np.clip(
            instance_coords[..., 1::2] - top, a_min=0, a_max=top + height
        )
        Data["instance_coords"] = instance_coords

    return Data


def resizeFn(
    Data: Dict,
    Size: Union[Sequence, int],
    Interpolation=T.InterpolationMode.BILINEAR,
) -> Dict:
    Img = Data.pop("image")
    
    W, H = F.get_image_size(Img) # W, H for Pillow image
    SizeH, SizeW = pair(Size) # B C H W for pytorch tensor
    
    # return if does not resize
    if (W, H) == (SizeW, SizeH):
        Data["image"] = Img
        return Data
    
    Data["image"] = F.resize(Img, pair(Size), Interpolation)
    
    if "mask" in Data:
        Mask = Data.pop("mask")
        ResizedMask = F.resize(Mask, pair(Size), T.InterpolationMode.NEAREST)
        Data["mask"] = ResizedMask

    if "box_coordinates" in Data:
        Boxes = Data.pop("box_coordinates")
        Boxes[:, 0::2] *= 1.0 * SizeW / W
        Boxes[:, 1::2] *= 1.0 * SizeH / H
        Data["box_coordinates"] = Boxes

    if "instance_mask" in Data:
        assert "instance_coords" in Data

        InsMasks = Data.pop("instance_mask")

        ResizedInsMasks = F.resize(InsMasks, pair(Size), T.InterpolationMode.NEAREST)
        Data["instance_mask"] = ResizedInsMasks

        InsCoords = Data.pop("instance_coords")
        InsCoords = InsCoords.astype(np.float)
        InsCoords[..., 0::2] *= 1.0 * SizeW / W
        InsCoords[..., 1::2] *= 1.0 * SizeH / H
        Data["instance_coords"] = InsCoords
    return Data


class SegDetToTensor(T.ToTensor):
    def __init__(self) -> None:
        super().__init__()
    def __call__(self, Data: Dict) -> Dict:
        # HWC --> CHW
        Img = Data["image"]
        Data["image"] = F.to_tensor(Img)

        if "mask" in Data:
            Mask = Data.pop("mask")
            Mask = np.array(Mask)
            Data["mask"] = torch.as_tensor(Mask, dtype=torch.long)

        if "box_coordinates" in Data:
            Boxes = Data.pop("box_coordinates")
            Data["box_coordinates"] = torch.as_tensor(Boxes, dtype=torch.float)

        if "box_labels" in Data:
            BoxesLabels = Data.pop("box_labels")
            Data["box_labels"] = torch.as_tensor(BoxesLabels)

        if "instance_mask" in Data:
            assert "instance_coords" in Data
            InsMasks = Data.pop("instance_mask")
            Data["instance_mask"] = InsMasks.to(dtype=torch.long)

            InsCoords = Data.pop("instance_coords")
            Data["instance_coords"] = torch.as_tensor(InsCoords, dtype=torch.float)
        return Data


class SegDetNormalisation(T.Normalize):
    def __init__(self, mean, std, inplace=False):
        super().__init__(mean, std, inplace)
    def forward(self, Data: Dict) -> Dict:
        Img = Data["image"]
        Data["image"] = F.normalize(Img, self.mean, self.std, self.inplace)
        return Data

    
class SegRandomShortSizeResize(object):
    """
    This class implements random resizing such that shortest side is between specified minimum and maximum values.
    """
    def __init__(
        self, 
        MinShortSide=256, 
        MaxShortSide=768, 
        DimMaxImg=1024, 
        Interpolation=F.InterpolationMode.BILINEAR
        ) -> None:
        super().__init__()
        self.MinShortSide = MinShortSide
        self.MaxShortSide = MaxShortSide
        self.DimMaxImg = DimMaxImg
        self.Interpolation = Interpolation

    def __call__(self, Data: Dict) -> Dict:
        ShortSide = random.randint(self.MinShortSide, self.MaxShortSide)
        W, H = Data["image"].size
        Scale = min(
            ShortSide / min(H, W), self.DimMaxImg / max(H, W)
        )
        W = int(W * Scale)
        H = int(H * Scale)
        Data = resizeFn(Data, Size=(H, W), Interpolation=self.Interpolation)
        return Data


class SegDetResize(T.Resize):
    def __init__(self, size, Interpolation=F.InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__(size, Interpolation, max_size, antialias)
        self.Interpolation = Interpolation
    def forward(self, Data: Dict) -> Dict:
        return resizeFn(Data, Size=self.size, Interpolation=self.Interpolation)


class SegDetRandomHorizontalFlip(T.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)
    def forward(self, Data: Dict) -> Dict:
        if torch.rand(1) < self.p:
            Img = Data["image"]
            W, _ = F.get_image_size(Img)
            Data["image"] = F.hflip(Img)
            
            if "mask" in Data:
                Mask = Data.pop("mask")
                Data["mask"] = F.hflip(Mask)

            if "box_coordinates" in Data:
                Boxes = Data.pop("box_coordinates")
                Boxes[..., 0::2] = W - Boxes[..., 2::-2]
                Data["box_coordinates"] = Boxes

            if "instance_mask" in Data:
                assert "instance_coords" in Data

                InsCoords = Data.pop("instance_coords")
                InsCoords[..., 0::2] = W - InsCoords[..., 2::-2]
                Data["instance_coords"] = InsCoords

                InsMasks = Data.pop("instance_mask")
                Data["instance_mask"] = F.hflip(InsMasks)
        return Data


class SegRandomGaussianBlur(object):
    def __init__(self, p=0.5, **kwargs):
        super().__init__()
        self.p = p

    def __call__(self, Data: Dict) -> Dict:
        if random.random() < self.p:
            Img = Data.pop("image")
            # radius is the standard devaition of the gaussian kernel
            Img = Img.filter(ImageFilter.GaussianBlur(radius=random.random()))
            Data["image"] = Img
        return Data


class SegRandomRotation(T.RandomRotation):
    # waite for box rotation
    def __init__(self, degrees, Interpolation=F.InterpolationMode.BILINEAR, expand=False, center=None, fill=0, resample=None):
        super().__init__(degrees, Interpolation, expand, center, fill, resample)
    def rotateProcess(self, Img, Angle):
        Fill = self.fill
        # Channels, _, _ = F.get_dimensions(Img)
        Channels = len(Img.mode)
        if isinstance(Img, Tensor):
            if isinstance(Fill, (int, float)):
                Fill = [float(Fill)] * Channels
            else:
                Fill = [float(f) for f in Fill]
        return F.rotate(Img, Angle, self.resample, self.expand, self.center, Fill)

    def forward(self, Data):
        Angle = self.get_params(self.degrees)
        
        Img = Data['image']
        Data['image'] = self.rotateProcess(Img, Angle)
        
        Mask = Data['mask']
        Data['mask'] = self.rotateProcess(Mask, Angle)
        return Data


class SegDetRandomCrop(object):
    """
    This method randomly crops an image area.

    .. note::
        If the size of input image is smaller than the desired crop size, the input image is first resized
        while maintaining the aspect ratio and then cropping is performed.
    """

    def __init__(
        self,
        size: Union[Sequence, int],
        fill: Optional[int]=0,
        ignore_idx: Optional[int]=0,
        **kwargs
    ) -> None:
        super().__init__()
        self.SizeH, self.SizeW = pair(size)
        self.seg_class_max_ratio = 0.75
        self.ignore_idx = ignore_idx
        self.num_repeats = 10
        self.seg_fill = fill
        pad_if_needed = True
        self.if_needed_fn = (
            self._pad_if_needed if pad_if_needed else self._resize_if_needed
        )

    @staticmethod
    def get_params(img_h, img_w, target_h, target_w):
        if img_w == target_w and img_h == target_h:
            return 0, 0, img_h, img_w

        i = random.randint(0, max(0, img_h - target_h))
        j = random.randint(0, max(0, img_w - target_w))
        return i, j, target_h, target_w

    @staticmethod
    def get_params_from_box(boxes, img_h, img_w):
        # x, y, w, h
        offset = random.randint(20, 50)
        start_x = max(0, int(round(np.min(boxes[..., 0]))) - offset)
        start_y = max(0, int(round(np.min(boxes[..., 1]))) - offset)
        end_x = min(int(round(np.max(boxes[..., 2]))) + offset, img_w)
        end_y = min(int(round(np.max(boxes[..., 3]))) + offset, img_h)

        return start_y, start_x, end_y - start_y, end_x - start_x

    def get_params_from_mask(self, Data, i, j, h, w):
        img_w, img_h = F.get_image_size(Data["image"])
        for _ in range(self.num_repeats):
            temp_data = cropFn(
                copy.deepcopy(Data), top=i, left=j, height=h, width=w
            )
            class_labels, cls_count = np.unique(
                np.array(temp_data["mask"]), return_counts=True
            )
            valid_cls_count = cls_count[class_labels != self.ignore_idx]

            if valid_cls_count.size == 0:
                continue

            # compute the ratio of segmentation class with max. pixels to total pixels.
            # If the ratio is less than seg_class_max_ratio, then exit the loop
            total_valid_pixels = np.sum(valid_cls_count)
            max_valid_pixels = np.max(valid_cls_count)
            ratio = max_valid_pixels / total_valid_pixels

            if len(cls_count) > 1 and ratio < self.seg_class_max_ratio:
                break
            i, j, h, w = self.get_params(
                img_h=img_h, img_w=img_w, target_h=self.SizeH, target_w=self.SizeW
            )
        return i, j, h, w

    def _resize_if_needed(self, Data: Dict) -> Dict:
        Img = Data["image"]

        w, h = F.get_image_size(Img)
        # resize while maintaining the aspect ratio
        new_size = min(h + max(0, self.SizeH - h), w + max(0, self.SizeW - w))

        return resizeFn(
            Data, Size=new_size, Interpolation=T.InterpolationMode.BILINEAR
        )

    def _pad_if_needed(self, Data: Dict) -> Dict:
        Img = Data.pop("image")

        w, h = F.get_image_size(Img)
        new_w = w + max(self.SizeW - w, 0)
        new_h = h + max(self.SizeH - h, 0)

        pad_img = Image.new(Img.mode, (new_w, new_h), color=0)
        pad_img.paste(Img, (0, 0))
        Data["image"] = pad_img

        if "mask" in Data:
            mask = Data.pop("mask")
            pad_mask = Image.new(mask.mode, (new_w, new_h), color=self.seg_fill)
            pad_mask.paste(mask, (0, 0))
            Data["mask"] = pad_mask

        return Data

    def __call__(self, Data: Dict) -> Dict:
        # box_info
        if "box_coordinates" in Data:
            boxes = Data.get("box_coordinates")
            # crop the relevant area
            image_w, image_h = F.get_image_size(Data["image"])
            box_i, box_j, box_h, box_w = self.get_params_from_box(
                boxes, image_h, image_w
            )
            Data = cropFn(Data, top=box_i, left=box_j, height=box_h, width=box_w)

        Data = self.if_needed_fn(Data)

        img_w, img_h = F.get_image_size(Data["image"])
        i, j, h, w = self.get_params(
            img_h=img_h, img_w=img_w, target_h=self.SizeH, target_w=self.SizeW
        )

        if (
            "mask" in Data
            and self.seg_class_max_ratio is not None
            and self.seg_class_max_ratio < 1.0
        ):
            i, j, h, w = self.get_params_from_mask(Data=Data, i=i, j=j, h=h, w=w)

        Data = cropFn(Data, top=i, left=j, height=h, width=w)
        return Data

    def __repr__(self) -> str:
        return "{}(size=(h={}, w={}), seg_class_max_ratio={}, seg_fill={})".format(
            self.__class__.__name__,
            self.SizeH,
            self.SizeW,
            self.seg_class_max_ratio,
            self.seg_fill,
        )
    

class RandomOrder(object):
    """
    This method applies a list of all or few transforms in a random order.
    """
    def __init__(self, AugList: List, KFactor=1.0) -> None:
        super().__init__()
        self.AugList = AugList
        assert (
            0.0 < KFactor <= 1.0
        ), "--image-augmentation.random-order.apply-k should be > 0 and <= 1"
        self.KeepT = int(math.ceil(len(self.AugList) * KFactor))

    def __call__(self, Data: Dict) -> Dict:
        random.shuffle(self.AugList)
        for t in self.AugList[: self.KeepT]:
            Data = t(Data)
        return Data
    
    
class PhotometricDistort(object):
    """
    This class implements Photometeric distorion.

    .. note::
        Hyper-parameters of PhotoMetricDistort in PIL and OpenCV are different. Be careful
    """

    def __init__(self, p=0.5) -> None:
        # Contrast
        alpha_min = 0.5
        alpha_max = 1.5
        
        Contrast = T.ColorJitter(contrast=[alpha_min, alpha_max])

        # Brightness
        beta_min = 0.875
        beta_max = 1.125
        Brightness = T.ColorJitter(brightness=[beta_min, beta_max])

        # Saturation
        gamma_min = 0.5
        gamma_max = 1.5
        Saturation = T.ColorJitter(saturation=[gamma_min, gamma_max])

        # Hue
        delta_min = -0.05
        delta_max = 0.05
        Hue = T.ColorJitter(hue=[delta_min, delta_max])

        super().__init__()
        self.Brightness = Brightness
        self.Contrast = Contrast
        self.Hue = Hue
        self.Saturation = Saturation
        self.p = p
    
    def applyTransformations(self, Img):
        r = np.random.rand(7)

        if r[0] < self.p:
            Img = self.Brightness(Img)

        contrast_before = r[1] < self.p
        if contrast_before and r[2] < self.p:
            Img = self.Contrast(Img)

        if r[3] < self.p:
            Img = self.Saturation(Img)

        if r[4] < self.p:
            Img = self.Hue(Img)

        if not contrast_before and r[5] < self.p:
            Img = self.Contrast(Img)

        if r[6] < self.p and Img.mode != "L":
            # Only permute channels for RGB images
            # [H, W, C] format
            NpImg = np.asarray(Img)
            NumChannels = NpImg.shape[2]
            NpImg = NpImg[..., np.random.permutation(range(NumChannels))]
            Img = Image.fromarray(NpImg)
        return Img

    def __call__(self, Data: Dict) -> Dict:
        Img = Data.pop("image")
        Data["image"] = self.applyTransformations(Img)
        return Data


def intersect(box_a, box_b):
    """Computes the intersection between box_a and box_b"""
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccardNumpy(box_a: np.ndarray, box_b: np.ndarray):
    """
    Computes the intersection of two boxes.
    Args:
        box_a (np.ndarray): Boxes of shape [Num_boxes_A, 4]
        box_b (np.ndarray): Box osf shape [Num_boxes_B, 4]

    Returns:
        intersection over union scores. Shape is [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])  # [A,B]
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarization(object):
    """ Apply Solarization to the PIL image. """
    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            return ImageOps.solarize(x)
        else:
            return x


def randomMaskingPytorch(x: Tensor, MaskRatio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [B, C, H, W], sequence
    """
    C, H, W = x.shape

    XMaksed = x.reshape(C, H * W)
    C, L = XMaksed.shape  # batch, length, dim
    LenKeep = int(L * (1 - MaskRatio))

    # sort noise for each sample, 
    # x is not shuflle, just get random index
    # ascend: small is keep, large is remove
    Noise = torch.rand(L, device=x.device)  # noise in [0, 1]
    IdxShuffle = torch.argsort(Noise, dim=0)
    
    # generate the binary mask: 1 is keep, 0 is remove
    Mask = torch.zeros([L], device=x.device)
    Mask[:LenKeep] = 1
    Mask = torch.gather(Mask, dim=0, index=IdxShuffle)
    Mask = Mask.reshape(H, W).unsqueeze(0)

    return x.masked_fill(Mask==0, 0), Mask


def randomMaskingPil(x: Image.Image, MaskRatio):
    W, H = x.size

    L = H * W
    LenKeep = int(L * (1 - MaskRatio))

    # sort noise for each sample, 
    # x is not shuflle, just get random index
    # ascend: small is keep, large is remove
    Noise = np.random.rand(L)  # noise in [0, 1]
    IdxShuffle = np.argsort(Noise, axis=0)

    # generate the binary mask: 0 is keep, 1 is remove
    NpMask = np.ones([L])
    NpMask[:LenKeep] = 0
    NpMask = NpMask[IdxShuffle]
    NpMask = NpMask.reshape((H, W))
    
    Mask = Image.fromarray(NpMask == 1)
    BlackImg = Image.new(x.mode, (W, H))
    x.paste(BlackImg, (0, 0), Mask)

    return x, Mask


class RandomMasking(object):
    """ Apply random mask to the PIL image. 
    Effect like Guassian blur
    """
    def __init__(self, MaskRatio):
        self.MaskRatio = MaskRatio

    def __call__(self, Data):
        if self.MaskRatio is not None and 0 < self.MaskRatio < 1:
            if isinstance(Data, Dict):
                Img = Data.pop("image")
            else:
                Img = Data
                
            if isinstance(Img, Tensor):
                OutImg, _ = randomMaskingPytorch(Img, self.MaskRatio) # after to tensor
            else:
                OutImg, _ = randomMaskingPil(Img, self.MaskRatio) # before to tensor
              
            if isinstance(Data, Dict):
                Data["image"] = OutImg
            else:
                Data = OutImg
            
        return Data


MixMethodRegistry = {}


def registerMixMethod(Name):
    def registerMethodClass(Cls):
        if Name in MixMethodRegistry:
            raise ValueError("Cannot register duplicate model ({})".format(Name))
        MixMethodRegistry[Name] = Cls
        return Cls
    return registerMethodClass


def getMixMethod(MethodName):
    MixMethod = None
    if MethodName in MixMethodRegistry:
        MixMethod = MixMethodRegistry[MethodName]
    else:
        SupportedMethods = list(MixMethodRegistry.keys())
        SuppStr = "Supported methods are:"
        for i, Name in enumerate(SupportedMethods):
            SuppStr += "\n\t {}: {}".format(i, colorText(Name))
    return MixMethod


def correctLam(ImgShape, BBox):
    W, H = ImgShape
    Upper, Lower, Left, Right = BBox
    BBoxArea = (Lower - Upper) * (Right - Left) 
    MixLam = 1. - BBoxArea / float(W * H)
    return MixLam


def correctCircleLam(ImgShape, BBox):
    W, H = ImgShape
    Upper, Lower, Left, Right = BBox
    BBoxArea = (Lower - Upper) * (Right - Left) 
    MixLam = 4 * ((W * H) - BBoxArea) / (math.pi * (min(W, H) ** 2))
    return MixLam
    
    
def randBBox(ImgShape, Ratio, Margin=0., Count=None):
    W, H = ImgShape
    HCut, WCut = int(H * Ratio), int(W * Ratio)
    YMargin, XMargin = int(Margin * HCut), int(Margin * WCut)
    Cy = np.random.randint(0 + YMargin, H - YMargin, size=Count)
    Cx = np.random.randint(0 + XMargin, W - XMargin, size=Count)
    Upper = np.clip(Cy - HCut // 2, 0, H) # Cy - HCut // 2 is between (0, H)
    Lower = np.clip(Cy + HCut // 2, 0, H)
    Left = np.clip(Cx - WCut // 2, 0, W)
    Right = np.clip(Cx + WCut // 2, 0, W)
    return Upper, Lower, Left, Right


def randWRBBox(ImgShape, Ratio):
    W, H = ImgShape
    HCropLen, WCropLen = int(H * Ratio), int(W * Ratio)
    
    Left = (W - WCropLen) / 2
    Upper = (H - HCropLen) / 2
    Right = Left + WCropLen
    Lower = Upper + HCropLen
    
    DeltaX = random.uniform(- Left, Left)
    DeltaY = random.uniform(- Upper, Upper)

    Left = round(Left + DeltaX)
    Upper = round(Upper + DeltaY)
    Right = round(Right + DeltaX)
    Lower = round(Lower + DeltaY)
    
    return Upper, Lower, Left, Right


def cutmixBBox(ImgShape, Lambda, Margin=0., Count=None):
    """ Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border Margin as percent of bbox dimensions.

    Args:
        ImgShape (tuple): Image shape as tuple
        Lambda (float): Cutmix lambda value
        Margin (float): Percentage of bbox dimension to enforce as Margin (reduce amount of box outside image)
        Count (int): Number of bbox to generate
    """
    return randBBox(ImgShape, np.sqrt(1 - Lambda), Margin, Count)


def wrmixBBox(ImgShape, Lambda, MinCropRatio):
    Lambda = MinCropRatio + ((1 - Lambda) ** (0.25)) * (1 - MinCropRatio)
    Upper, Lower, Left, Right = randWRBBox(ImgShape, Lambda)
    BBox = [Upper, Lower, Left, Right]
    return BBox, correctLam(ImgShape, BBox) # two mask are relative to each other


def hmixBbox(ImgShape, Lambda, HMixRatio, Margin=0, Count=None):
    """ Generate hmix bbox 
    https://github.com/naver-ai/hmix-gmix/blob/main/mixup.py
    """
    Upper, Lower, Left, Right = \
        randBBox(ImgShape, np.sqrt(1 - Lambda) * np.sqrt(HMixRatio), Margin, Count)
        
    BBox = [Upper, Lower, Left, Right]
    return BBox, correctLam(ImgShape, BBox)


def gmixGaussianPil(ImgShape, Lambda):
    W, H = ImgShape
    Lambda = 1 - Lambda

    # select random center point 
    Cx = np.random.randint(0, W)
    Cy = np.random.randint(0, H)

    XGrid = (np.expand_dims((np.arange(W) - Cx), axis=1)).repeat(H, axis=1)
    YGrid = (np.expand_dims((np.arange(H) - Cy), axis=1)).repeat(W, axis=1).transpose()

    Grid = np.stack([XGrid, YGrid], axis=-1) ** 2.
    Dist = np.sum(Grid, axis=-1)

    GMixMask = np.exp(- Dist * math.pi / (Lambda * H * W))
    # # wrong code, wrong position:
    # GMixMask = 1. - np.exp(- Dist * math.pi / (2 * Lambda * H * W))

    GMixLam = GMixMask.sum() / (H * W)
    return GMixMask, GMixLam


def gmixGaussianPytorch(ImgShape, Lambda):
    """ Generate gmix kernel 
    https://github.com/naver-ai/hmix-gmix/blob/main/mixup.py
    """
    W, H = ImgShape
    Lambda = 1 - Lambda

    # select random center point 
    Cx = np.random.randint(0, W)
    Cy = np.random.randint(0, H)

    XGrid = (torch.arange(W) - Cx).repeat(H, 1)
    YGrid = (torch.arange(H) - Cy).repeat(W, 1).t()

    Grid = torch.stack([XGrid, YGrid], dim=-1) ** 2.
    Dist = torch.sum(Grid, dim=-1)
    GMixMask = torch.exp(- Dist * math.pi / (Lambda * H * W))
    # GMixMask = 1. - torch.exp(- Dist * math.pi / (2 * Lambda * H * W))

    GMixLam = GMixMask.sum() / (H * W)
    return GMixMask, GMixLam


@registerMixMethod("mixup")
def mixUp(Img1, Img2, Lambda, **kwargs: Any):
    '''Reference:
    mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
    '''
    return Image.blend(Img1, Img2, Lambda)


@registerMixMethod("cutmix")
def cutMixPil(Img1: Image.Image, Img2: Image.Image, Lambda, **kwargs: Any):
    W, H = Img1.size
    Upper, Lower, Left, Right = cutmixBBox([W, H], Lambda)
    NpMask = np.zeros([H, W]) # W x H to H x W
    NpMask[Upper:Lower, Left:Right] = 1
    Mask = Image.fromarray(NpMask == 1)
    Img1.paste(Img2, (0, 0), Mask)
    return Img1


@registerMixMethod("wrmix")
def WeightedRegionMixPil(Img1: Image.Image, Img2: Image.Image, 
               Alpha, Lambda, BaseBias, MinCropRatio,
               MaskRot, **kwargs: Any):
    W, H = Img1.size
    (Upper, Lower, Left, Right), NonMixRatio = wrmixBBox([W, H], Lambda, MinCropRatio)
    
    LambdaBias = BaseBias * NonMixRatio
    MixLam = max(min(np.random.beta(Alpha, Alpha) + LambdaBias, 1), 0)
    # MixLam = max(MixLam, 1 - MixLam)
    
    NpMask = np.zeros([H, W]) # W x H to H x W
    # NpMask[Upper:Lower, Left:Right] = MixLam if NonMixRatio > 0.5 else (1 - MixLam)
    NpMask[Upper:Lower, Left:Right] = 1 - MixLam # always NonMixRatio lower than 0.5

    if MaskRot:
        Mask = Image.fromarray(NpMask)
        Mask = Mask.rotate(random.uniform(0, 360), Image.BILINEAR, fillcolor=0)
        NpMask = np.array(Mask)
    # b = Image.fromarray(NpMask==1 - MixLam)
    # b.show()
    NpMask = np.expand_dims(NpMask, axis=-1)
    # a = Image.fromarray((NpMask * np.array(Img1) + (1. - NpMask) * np.array(Img2)).astype(np.uint8))
    # a.show()
    return Image.fromarray((NpMask * np.array(Img1) + (1. - NpMask) * np.array(Img2)).astype(np.uint8))


@registerMixMethod("hmix")
def hMixPil(Img1: Image.Image, Img2: Image.Image, Lambda, HMixRatio, **kwargs: Any):
    W, H = Img1.size
    # Larger the cut box, lower the NonMixRatio, so larger the MixLam
    (Upper, Lower, Left, Right), NonMixRatio = hmixBbox([W, H], Lambda, HMixRatio)
    MixLam = Lambda / NonMixRatio 
    # MixupLambda = Lambda / (1. - (1. - Lambda) * HMixRatio)
    NpMask = np.zeros([H, W]) # W x H to H x W
    NpMask[Upper:Lower, Left:Right] = 1
    Mask = Image.fromarray(NpMask == 1)
    Img1.paste(Img2, (0, 0), Mask)
    return Image.blend(Img1, Img2, 1 - MixLam)


@registerMixMethod("gmix")
def gMixPil(Img1: Image.Image, Img2: Image.Image, Lambda, **kwargs: Any):
    W, H = Img1.size
    NpMask, GMixLam = gmixGaussianPil([W, H], Lambda)
    NpMask = np.expand_dims(NpMask, axis=-1)
    return Image.fromarray((NpMask * np.array(Img1) + (1. - NpMask) * np.array(Img2)).astype(np.uint8))


def getMixMethodWithHybrid(MethodName):
    MixMethodPool = list(MixMethodRegistry.keys())
        
    if 'hybrid' in MethodName:
        mixMethod = getMixMethod(MixMethodPool[np.random.randint(len(MixMethodPool))])
    else:
        mixMethod = getMixMethod(MethodName)
        
    return mixMethod


def adaptSizeMix(opt, SupImg1: Image.Image, SupImg2: Image.Image):
    if SupImg1.size != SupImg2.size:
        W1, H1 = SupImg1.size
        W2, H2 = SupImg2.size
        
        MinW = min(W1, W2)
        MinH = min(H1, H2)

        Left1 = random.uniform(0, W1 - MinW)
        Left2 = random.uniform(0, W2 - MinW)
        Upper1 = random.uniform(0, H1 - MinH)
        Upper2 = random.uniform(0, H2 - MinH)

        SupImg1 = SupImg1.crop([Left1, Upper1, Left1 + MinW, Upper1 + MinH])
        SupImg2 = SupImg2.crop([Left2, Upper2, Left2 + MinW, Upper2 + MinH])
    
    mixMethod = getMixMethodWithHybrid(opt.mix_method)
    Lambda = np.random.beta(opt.mix_alpha, opt.mix_alpha) if opt.mix_alpha > 0 else 0.5
    
    if random.random() < 0.5:
        MixImg = mixMethod(SupImg1, SupImg2, Alpha=opt.mix_alpha, Lambda=Lambda, 
                           BaseBias=opt.mix_base_bias, MinCropRatio=opt.mix_mincrop_ratio,
                           HMixRatio=opt.hmix_ratio, MaskRot=opt.mix_maskrot)
    else:
        MixImg = mixMethod(SupImg2, SupImg1, Alpha=opt.mix_alpha, Lambda=Lambda, 
                           BaseBias=opt.mix_base_bias, MinCropRatio=opt.mix_mincrop_ratio,
                           HMixRatio=opt.hmix_ratio, MaskRot=opt.mix_maskrot)

    return MixImg
