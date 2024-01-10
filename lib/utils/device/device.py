from math import ceil
import multiprocessing

import torch
import torch.cuda as Cuda

from ..utils import colorText


Mode = 1 # 0, 1, 2
NUM_GPU = Cuda.device_count()
if Mode == 0:
    CUDA_AVAI = False
    DataParallel = False
    Device = torch.device('cpu')
elif Mode == 1:
    # GPU
    CUDA_AVAI = Cuda.is_available()
    DataParallel = NUM_GPU > 1
    Device = torch.device('cuda:0' if CUDA_AVAI else 'cpu')
    # Device = torch.device('cuda:1' if CUDA_AVAI else 'cpu') # Multi-device


def workerManager(BatchSize, NumWorkers=None, IsPrint=True):
    NUM_PROC = multiprocessing.cpu_count()
    if not NumWorkers:
        if CUDA_AVAI:
            if BatchSize <= 48:
                WorkerKanban = BatchSize // 2
            elif BatchSize <= 64:
                WorkerKanban = BatchSize // 4
            else:
                WorkerKanban = BatchSize // 8
            NumWorkers = ceil(min(WorkerKanban, NUM_PROC - 2) / 2.) * 2
        else:
            NumWorkers = min(ceil(min(4 * round(NUM_PROC / 8), NUM_PROC - 2) / 2.) * 2, BatchSize)
            
    PinMemory = True if CUDA_AVAI else False
    if IsPrint:
        print("We use [%s/%d] workers, and pin memory is %s" \
            % (colorText(str(NumWorkers)), NUM_PROC, colorText(str(PinMemory))))
    return PinMemory, NumWorkers