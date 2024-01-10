import torch
import numpy as np
from typing import List, Dict

from . import registerCollateFn
from .defaultCollateFn import defaultCollateFn


def pretextProcess(Batch: List[Dict] or Dict):
    IsShuffle = False
    Label = Batch[0]["label"]
    if isinstance(Label, (int, float, np.int64)):
        NumViews = 1
    elif isinstance(Label, np.ndarray):
        NumViews = Label.size
    elif isinstance(Label, torch.Tensor):
        NumViews = Label.__len__() # for rotation, position etc.
    else:
        # list
        NumViews = len(Label)
    
    if NumViews == 1:
        return defaultCollateFn(Batch), IsShuffle
    else:
        IsShuffle = True
        
        Keys = Batch[0].keys() #list(Batch[0].keys())

        NewBatch = {k: [] for k in Keys}
        for b in Batch:
            for k in Keys:
                NewBatch[k].append(b[k])

        # stack the Keys
        for k in Keys:
            BatchElements = NewBatch.pop(k)
            
            try:
                BatchElements = torch.cat(BatchElements)
            except Exception as e:
                print("Unable to stack the tensors. Error: {}".format(e))
                    
            NewBatch[k] = BatchElements
        
    return NewBatch, IsShuffle


def batchShuffle(Batch):
    if not isinstance(Batch['image'], dict):
        IdxShuffle = torch.randperm(Batch['label'].shape[0], device=Batch['label'].device)
        for k in Batch.keys():
            Batch[k] = Batch[k][IdxShuffle]
    else:
        IdxShuffle = torch.randperm(Batch['image']['image'].shape[0], 
                                device=Batch['image']['image'].device)
        for hkey, Keys in zip(Batch.keys(), [Batch['image'].keys(), Batch['label'].keys()]):
            for k in Keys:
                Batch[hkey][k] = Batch[hkey][k][IdxShuffle]
            
    return Batch


@registerCollateFn("pretext")
def pretextCollateFn(Batch: List[Dict] or Dict):
    NewBatch, _ = pretextProcess(Batch)
    return NewBatch


@registerCollateFn("pretext_batch_shuffle")
def pretextBSCollateFn(Batch: List[Dict] or Dict):
    NewBatch, IsShuffle = pretextProcess(Batch)
    
    if IsShuffle: 
        NewBatch = batchShuffle(NewBatch)
        
    return NewBatch
