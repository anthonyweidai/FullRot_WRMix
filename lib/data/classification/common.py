import numpy as np
from typing import Any
from pathlib import Path

from ..baseDataset import ClsBaseDataset
from ..mydatasetReg import registerDataset


@registerDataset(Name="common", Task="classification")
class Mydataset(ClsBaseDataset):
    # For CPU dataloader
    def __init__(self, opt, ImgPaths, Transform=None, **kwargs: Any) -> None:
        super(Mydataset, self).__init__(opt, ImgPaths, Transform, **kwargs)
        self.getLabel()
    
    def getLabel(self):
        # Convert the data type of label to long int type
        self.Labels = np.zeros((len(self.ImgPaths), 1), dtype=np.int64)
        LabelInfos = [Path(ImgPath).parts[-2] if self.opt.get_path_mode == 3 else \
            Path(ImgPath).parts[-1] for ImgPath in self.ImgPaths]
        
        for i, LabelInfo in enumerate(LabelInfos):
            # train/val mode has problem, if class has train/val or is in setname
            for j, Name in enumerate(self.ClassNames):
                if Name in  LabelInfo:
                    self.Labels[i] = j
                    break

    def __getitem__(self, Index):
        ImgPath = self.ImgPaths[Index]
        Label = self.Labels[Index][0]
        Img = self.readImagePil(ImgPath)
        
        ImgData = self.Transform(Img)
        
        Data = {'image': ImgData, 'label': Label, 'sample_id': Index}

        return Data
    
    def getNumEachClass(self):
        NumEachClass = np.zeros((self.NumClasses, 1)) # prevent some class doen't has data
        Unique, Counts = np.unique(self.Labels, return_counts=True)
        ClassCount = dict(zip(Unique, Counts))
        for i in range(self.NumClasses):
            NumEachClass[i] = ClassCount[i]
        return NumEachClass
    
    def getSenFactor(self):
        # for data imbalanced dataset
        NumEachClass = self.getNumEachClass()
        Reciprocal = np.reciprocal(NumEachClass)
        return Reciprocal / max(Reciprocal)