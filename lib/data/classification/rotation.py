import random
import numpy as np
from typing import Any

import torch
from PIL import Image

from .utils import cropRot, cropResizeScale
from ..utils import adaptSizeMix
from ..baseDataset import ClsBaseDataset
from ..mydatasetReg import registerDataset
from ...utils import colorText, makeDivisible, getLoaderBatch


def rotateImg(opt, Img: Image.Image, RotAngle):
    '''
    Sharp angle shape will be the presentations for recognition.
    Get the max inscribed circle instead.
    '''
    if opt.crop_mode == 0:
        OutImg =  Img.rotate(RotAngle, Image.BILINEAR, expand=True)
    else:
        OutImg = cropRot(opt, Img, RotAngle) 
        
    ResizeRes = opt.resize_res
    return cropResizeScale(OutImg, ResizeRes, CropResize=False)


@registerDataset("rotation", "classification")
class RotDataset(ClsBaseDataset):
    def __init__(self, opt, ImgPaths, Transform=None, **kwargs: Any) -> None:
        super().__init__(opt, ImgPaths, Transform, **kwargs)
        '''Get the ratio of circular area, rotate the part, cut the circular region'''

        self.opt = opt
        self.RotMultiplier = opt.rot_degree
        
        self.BaseDegrees = None
            
        self.WorkerBatch = getLoaderBatch(self.opt)
        
        print('The amount of rotation data: %s' \
            % colorText(str(self.__len__() * self.opt.views)))

    def callerInit(self): 
        self.IndexOrder = {} # achieve within one batch data mixture, one worker one batch

    def __getitem__(self, Index):
        if self.opt.mix_method is not None:
            # confirm that inter-images mixture within the same batch 
            WorkerInfo = torch.utils.data.get_worker_info()
            WorkerKey = 'w%d' % (WorkerInfo.id)
            if WorkerKey in self.IndexOrder.keys():
                self.IndexOrder[WorkerKey].append(Index)
            else:
                self.IndexOrder.update({WorkerKey: [Index]})
                
        ImgPath = self.ImgPaths[Index]
 
        ImgData = []
        RandIdxList = random.sample(range(0, self.NumClasses), self.opt.views)  
        
        for i in range(self.opt.views):
            Img = self.readImagePil(ImgPath)
            RotAngle = self.BaseDegrees[RandIdxList[i]] if np.any(self.BaseDegrees) \
                else RandIdxList[i] * self.RotMultiplier
            
            SupImg = rotateImg(self.opt, Img, RotAngle)
            
            if self.opt.mix_method is not None and np.random.rand() < self.opt.mix_gamma:
                if np.random.rand() < self.opt.mix_kappa:
                    # intra image mixup
                    SupIdx = Index
                else:
                    # inter image mixup
                    IndexOrder = self.IndexOrder[WorkerKey]
                    Round = len(IndexOrder) // self.WorkerBatch
                    RoundIdx = len(IndexOrder) % self.WorkerBatch
                    if RoundIdx == 1:
                        # first one of the batch does not have intra related image
                        SupIdx = Index
                    else:
                        while 1:
                            RandomIdx = random.randint(0, RoundIdx - 1) if RoundIdx > 0 \
                                else - random.randint(1, self.WorkerBatch)
                            SupIdx = Round * self.WorkerBatch + RandomIdx    
                            if IndexOrder[SupIdx] != Index:
                                break 

                SupImg2 = self.readImagePil(self.ImgPaths[SupIdx])
                SupImg2 = rotateImg(self.opt, SupImg2, RotAngle)
                
                SupImg = adaptSizeMix(self.opt, SupImg, SupImg2)
            
            ImgData.extend([self.Transform(SupImg)])
            
        if self.opt.views == 1:
            ImgData = ImgData[0]
            Label = RandIdxList[0] # np.asarray(RandIdxList[0], dtype=np.int64)
            SampleIdx = Index
        else:
            ImgData = torch.stack(ImgData)
            Label = torch.LongTensor(RandIdxList)
            SampleIdx = torch.zeros(self.opt.views, dtype=torch.long)
            SampleIdx.fill_(Index)
        
        Data = {"image": ImgData, "label": Label, "sample_id": SampleIdx}
        return Data
    
    def getNumEachClass(self):
        NumEachClass = np.zeros((self.NumClasses, 1), np.int64)
        NumEachClass.fill(self.__len__() * self.opt.views // self.NumClasses)
        return NumEachClass
