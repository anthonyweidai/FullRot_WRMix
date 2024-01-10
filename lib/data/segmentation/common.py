from pathlib import Path
from typing import Any, Dict

from .utils import colorPalettePASCAL
from ..mydatasetReg import registerDataset
from ..baseDataset import SegBaseDataset


@registerDataset("common", "segmentation")
@registerDataset("pascal", "segmentation")
@registerDataset("ade20k", "segmentation")
class PascalVOCDataset(SegBaseDataset):
    """
    Dataset class for the PASCAL VOC 2012 dataset
    The structure of PASCAL VOC dataset should be something like this:
        PASCALVOC/mask
        PASCALVOC/train
        PASCALVOC/val
    """
    def __init__(self, opt, ImgPaths, Transform=None, **kwargs: Any) -> None:
        super().__init__(opt, ImgPaths, Transform, **kwargs)
        self.opt = opt
        
        # _, self.Palette, self.Animal2OriIdx, self.Ori2AnimalIdx = colorPalettePASCAL(opt.setname)

    def getMask(self, ImgPath):
        if 'pascal' in self.opt.setname:
            MaksFolder = 'mask'
            Ext = 'png'
        else:
            MaksFolder = 'mask'
            Ext = 'png'
        ImgName = Path(ImgPath).stem
        MaskPath = "%s/%s/%s.%s" % (self.opt.dataset_path, MaksFolder, ImgName, Ext)
        return MaskPath
        
    def __getitem__(self, Index) -> Dict:
        ImgPath = self.ImgPaths[Index]
        Img = self.readImagePil(ImgPath)
        Mask = self.readMaskPil(self.getMask(ImgPath))
        
        Data = {'image': Img, 'mask': Mask, 'sample_id': Index}

        Data = self.Transform(Data)
        
        Mask = Data.pop("mask")
        Data['label'] = Mask
        return Data
