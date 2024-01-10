import os
import numpy as np
import pandas as pd
from typing import Optional, Any

import torch
from torch.utils import data
from torchvision import transforms
import torchvision.transforms.functional as F

from .utils import (
    SegRandomShortSizeResize, SegDetRandomHorizontalFlip,
    SegDetRandomCrop, SegRandomGaussianBlur, PhotometricDistort,
    SegRandomRotation, RandomOrder, SegDetToTensor, SegDetResize, 
    SegDetNormalisation
)

from ..utils import (pair, colorText, readImagePil, readMaskPil, listPrinter, makeDivisible)


class CustomDataset(data.Dataset):
    def __init__(self, opt, ImgPaths, Transform=None, IsTraining: Optional[bool]=True, 
                 TargetSet=None, **kwargs: Any) -> None:
        super().__init__()
        self.opt = opt
        self.ImgPaths =  np.asarray(ImgPaths)
        self.TargetSet = np.asarray(TargetSet) if TargetSet is not None else TargetSet
        self.ClassNames = opt.class_names
        self.IsTraining = IsTraining
        
        '''
        https://stats.stackexchange.com/questions/202287/why-standardization-of-the-testing-set-has-to-be-performed-with-the-mean-and-sd
        use only meanstd of train set, test meanstd vary from task
        Mean and std should be the same for train and test dataset
        '''
        self.MeanStdType = 'train' 
        self.MeanStdDP = 6 # DecimalPlaces
        
        self.readImagePil = readImagePil
        self.readMaskPil = readMaskPil
        
        self.initMeanStd()
        
        self.InterpolationMode = F.InterpolationMode.BICUBIC
        if not Transform:
            self.Transform = self.buildTransforms()

        print('The amount of original %s data: %s' % \
            ('train' if self.IsTraining else 'test', 
             colorText(str(self.__len__()))))
    
    def __getitem__(self, Index):
        pass

    def __len__(self):
        return len(self.ImgPaths)
    
    def getNumEachClass(self):
        pass
    
    def getSenFactor(self):
        pass

    def callerInit(self):
        pass
    
    def buildTransforms(self):
        pass
    
    def initMeanStd(self):
        ''' Normalization helps get data within a range and reduces 
        the skewness which helps learn faster and better.
        '''
        self.MeanValues = None
        self.StdValues = None
        
        if self.opt.use_meanstd:
            Flag = 0
            DataPath = str(self.opt.dataset_path) # str(Path(self.TrainSet[0][0]).parents[1])
            if self.opt.num_split > 1:
                if os.path.isfile(DataPath + '/norm_mean.csv'):
                    Flag = 1
                    if 'common' not in self.opt.sup_method and self.opt.extend_val:
                        # self-sup dataset with split
                        self.MeanValues = np.zeros(3)
                        self.StdValues = np.zeros(3)
                        for i in range(self.opt.num_split):
                            CsvValues = pd.read_csv(DataPath + '/norm_s%d.csv' % (i + 1), index_col=0)
                            self.MeanValues += np.array(CsvValues[0:1].values.tolist()[0])
                            self.StdValues += np.array(CsvValues[1:2].values.tolist()[0])
                        # approximation, each split has the close number of images
                        self.MeanValues = list(self.MeanValues / self.opt.num_split)
                        self.StdValues = list(self.StdValues / self.opt.num_split)
                    else:
                        CsvValues = pd.read_csv(DataPath + '/norm_mean.csv', index_col=0)
                        self.MeanValues = CsvValues[self.opt.split:self.opt.split + 1].values.tolist()[0]
                        CsvValues = pd.read_csv(DataPath + '/norm_std.csv', index_col=0)
                        self.StdValues = CsvValues[self.opt.split:self.opt.split + 1].values.tolist()[0]
            else:
                if os.path.isfile('%s/norm_%s.csv' % (DataPath, self.MeanStdType)):
                    Flag = 1
                    CsvValues = pd.read_csv('%s/norm_%s.csv' % (DataPath, self.MeanStdType), index_col=0)
                    self.MeanValues = CsvValues[0:1].values.tolist()[0]
                    self.StdValues = CsvValues[1:2].values.tolist()[0]
            
            if self.MeanValues is not None and self.StdValues is not None:
                self.MeanValues = [round(v, self.MeanStdDP) for v in self.MeanValues]
                self.StdValues = [round(v, self.MeanStdDP) for v in self.StdValues]
                
            # Norm values printing
            if self.IsTraining and Flag:
                MeanPrint = ', '.join(map(str, self.MeanValues))
                StdPrint = ', '.join(map(str, self.StdValues))
                ListName = ['%s_Mean' % self.MeanStdType.capitalize(), '%s_Std' % self.MeanStdType.capitalize()]
                ListValue = [MeanPrint, StdPrint]
                listPrinter(ListName, ListValue)
        

class ClsBaseDataset(CustomDataset):
    def __init__(self, opt, ImgPaths, Transform=None, **kwargs: Any) -> None:
        super().__init__(opt, ImgPaths, Transform, **kwargs)
        self.NumClasses = opt.cls_num_classes

    def buildTransforms(self):
        if self.IsTraining:
            if "common" in self.opt.sup_method:
                AugResize = transforms.RandomResizedCrop(self.opt.resize_res, interpolation=self.InterpolationMode)
                AugCommon = [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10, interpolation=F.InterpolationMode.BILINEAR, expand=False), # 180
                    transforms.ToTensor(),
                    ]
            else:
                if self.opt.crop_mode != 0:
                    AugResize = transforms.RandomResizedCrop(self.opt.resize_res, interpolation=self.InterpolationMode)
                else:
                    AugResize = transforms.Resize(pair(self.opt.resize_res), interpolation=self.InterpolationMode)
                    
                AugCommon = [transforms.ToTensor()]     
        else:
            ResizeRes = makeDivisible(self.opt.resize_res * 1.15, 8)
            AugResize = transforms.Resize(pair(ResizeRes), interpolation=self.InterpolationMode)
            '''
            If size is a sequence like (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            '''
            AugCommon = [transforms.CenterCrop(pair(self.opt.resize_res)), transforms.ToTensor()] # val tunes test process 
        
        AugList = []
        AugList.append(AugResize)
        AugList.extend(AugCommon)
        
        if self.MeanValues:
            AugList.append(transforms.Normalize(self.MeanValues, self.StdValues))

        return transforms.Compose(AugList)
    
    
class SegBaseDataset(CustomDataset):
    def __init__(self, opt, ImgPaths, Transform=None, **kwargs: Any) -> None:
        super().__init__(opt, ImgPaths, Transform, **kwargs)
        self.NumClasses = opt.seg_num_classes

    @staticmethod
    def convertMask2Tensor(Mask):
        # convert to tensor
        Mask = np.array(Mask)
        if len(Mask.shape) > 2 and Mask.shape[-1] > 1:
            Mask = np.ascontiguousarray(Mask.transpose(2, 0, 1))
        return torch.as_tensor(Mask, dtype=torch.long)
    
    def buildTransforms(self):
        if self.IsTraining:
            if "common" in self.opt.sup_method:
                # ImgHeight, ImgWidth = pair(self.opt.resize_res)
                MaskFill = self.opt.mask_fill
                
                FirstAug = SegRandomShortSizeResize(256, 768, 1024, self.InterpolationMode)
                AugList = [
                    SegDetRandomHorizontalFlip(),
                    SegDetRandomCrop(self.opt.resize_res, MaskFill, self.opt.ignore_idx),
                    SegRandomGaussianBlur(),
                    PhotometricDistort(),
                    SegRandomRotation(10, fill=MaskFill),
                    ]
                if self.opt.random_aug_order:
                    AugList = [FirstAug, RandomOrder(AugList), SegDetToTensor()]
                else:
                    AugList.insert(0, FirstAug)
                    AugList.append(SegDetToTensor())
            else:
                AugList = [
                    SegDetResize(self.opt.resize_res, self.InterpolationMode), 
                    SegDetToTensor()
                    ]
        else:
            AugList = [
                SegDetResize(self.opt.resize_res, self.InterpolationMode), 
                SegDetToTensor()
                ]
            
        if self.MeanValues:
            AugList.append(SegDetNormalisation(self.MeanValues, self.StdValues))
 
        return transforms.Compose(AugList)
    
