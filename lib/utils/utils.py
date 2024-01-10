import os
import importlib
from pathlib import Path

import re
import csv
import math
import random
import numpy as np
from PIL import Image
from typing import Any, List, Dict, Optional, Union

import torch
from torch import optim
from torch import distributed as dist

from .variables import TextColors


def colorText(in_text: str, Mode=1, ColourName='') -> str:
    if ColourName:
        return TextColors[ColourName] + in_text + TextColors['end_colour']
    else:
        if Mode == 1:
            # cyan colour, params init/input params
            return TextColors['light_cyan'] + in_text + TextColors['end_colour']
        elif Mode == 2:
            # yellow colour, results output
            return TextColors['light_yellow'] + in_text + TextColors['end_colour']
        elif Mode == 3:
            # green colour, others
            return TextColors['light_green'] + in_text + TextColors['end_colour']


def seedSetting(RPMode, Seed=999):
    # Set random seed for reproducibility
    if RPMode:
        #manualSeed = random.randint(1, 10000) # use if you want new results
        print("Random Seed: ", Seed)
        np.random.seed(Seed)
        random.seed(Seed)
        torch.manual_seed(Seed)
    return


def importModule(CurrentPath, RelativePath, SubFold=''):
    # Automatically import the modules
    ModulesDir = os.path.dirname(CurrentPath) + SubFold
    if os.path.isdir(ModulesDir):
        for file in os.listdir(ModulesDir):
            path = os.path.join(ModulesDir, file)
            if (
                    not file.startswith("_")
                    and not file.startswith(".")
                    and (file.endswith(".py") or os.path.isdir(path))
            ):
                ModuleName = file[: file.find(".py")] if file.endswith(".py") else file
                _ = importlib.import_module(RelativePath + ModuleName)


def getSubdirectories(Dir):
    return [SubDir for SubDir in os.listdir(Dir)
            if os.path.isdir(os.path.join(Dir, SubDir))]


def expFolderCreator(ExpType, ExpLevel='', TargetExp=None, Mode=0):
    # Count the number of exsited experiments
    FolderPath = './exp/%s/%s' % (ExpType, ExpLevel) 
    Path(FolderPath).mkdir(parents=True, exist_ok=True)
    
    ExpList = getSubdirectories(FolderPath)
    if TargetExp:
        ExpCount = TargetExp
    else:
        if len(ExpList) == 0:
            ExpCount = 1
        else:
            MaxNum = 0
            for idx in range(len(ExpList)):
                NumStr = re.findall('\d+', ExpList[idx])
                if NumStr: # should not be empty
                    temp = int(NumStr[0]) + 1
                    if MaxNum < temp:
                        MaxNum = temp
            ExpCount = MaxNum if Mode == 0 else MaxNum - 1
    
    DestPath = '%s/exp%s/' % (FolderPath, str(ExpCount))
    Path(DestPath).mkdir(parents=True, exist_ok=True)
    Path(DestPath + '/model').mkdir(parents=True, exist_ok=True)
    
    return DestPath, ExpCount


def writeCsv(DestPath, FieldName, FileData, NewFieldNames=[], DictMode=False):
    Flag = 0 if os.path.isfile(DestPath) else 1
    
    with open(DestPath, 'a', encoding='UTF8', newline='') as f:
        if DictMode:
            writer = csv.DictWriter(f, fieldnames=FieldName)
            if Flag == 1:
                writer.writeheader()
            writer.writerows(FileData) # write data
        else:
            writer = csv.writer(f)
            if Flag == 1:
                if NewFieldNames != []:
                    _ = [FieldName.append(FiledName) for FiledName in NewFieldNames]
                writer.writerow(FieldName) # write the header
            writer.writerow(FileData) # write data


def pair(Res):
    return Res if isinstance(Res, tuple) else (Res, Res)


def makeDivisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.Py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def wightFrozen(Model, FreezeWeight, TransferFlag=0,  PreTrained=1):
    # ModelDict = Model.state_dict()
    if FreezeWeight == 0:
        print("Skip freezing layers")
        return Model
    else:
        Idx = 0
        for Name, Param in Model.named_parameters():
            if FreezeWeight == 1: 
                # if 'features' in Name:
                # Judger = 'classifier' not in Name.lower()
                # if PreTrained == 2:
                #     Judger = Judger and idx < len(ModelDict.keys()) - 7
                    
                if 'classifier' not in Name.lower():
                    Param.requires_grad = False
                else:
                    print(Name, Param.requires_grad)      
            elif FreezeWeight == 2:
                # Freeze the layers with transferred weight
                while 1:
                    if TransferFlag[Idx] == 1:
                        Param.requires_grad = False
                        break
                    elif TransferFlag[Idx] == 2:
                        Idx += 1
                    else:
                        print(Name, Param.requires_grad)
                        break
                Idx += 1
            elif FreezeWeight == 3:
                # For step weight freezing
                Param.requires_grad = True
            elif FreezeWeight == 4:
                Param.requires_grad = False
            else:
                print(Name, Param.requires_grad)
                
        if FreezeWeight == 3:
            print("Unfreeze all layers")
        elif FreezeWeight == 4:
            print("Freeze all layers")
            
        return Model


def ignoreTransfer(NewKey, IgnoreStrList, TransferFlag, IdxNew, FalseFlag=0):
    if any(s in NewKey for s in IgnoreStrList):
        TransferFlag[IdxNew] = 2
    else:
        TransferFlag[IdxNew] = FalseFlag
    return TransferFlag


def loadModelWeight(Model, FreezeWeight, PreTrainedWeight, 
                    PreTrained=1, DistMode=False, DropLast=False, SkipSELayer=False):
    '''
    cannot load pkl file, "Invalid magic number; corrupt file?"
    '''
    IgnoreStrList = ["running_mean", "running_var", "num_batches_tracked"] # pytorch 1.10
    
    print('Knowledge transfer from: %s' %(PreTrainedWeight))
    
    PretrainedDict = torch.load(PreTrainedWeight, map_location=torch.device('cpu'))
    
    ModelDict = Model.state_dict()
    TransferFlag = np.zeros((len(ModelDict), 1))
    if PreTrained == 1 or PreTrained == 3:
        # Get weight if pretrained weight has the same dict
        if PreTrained == 1:
            # PretrainedDictNew = {}
            # for Idx, (k, v) in enumerate(PretrainedDict.items()):
            #     if k in ModelDict and 'classifier' not in k.lower():
            #         Temp = {k: v}
            #         PretrainedDictNew.append(Temp)
            #         TransferFlag[Idx] = 1
            # PretrainedDict = PretrainedDictNew
            PretrainedDict = {k: v for k, v in PretrainedDict.items() if k in ModelDict and ('classifier' not in k.lower() or DistMode)}
        else:
            PretrainedDict = {k.replace('module.',''): v for k, v in PretrainedDict.items()}
            TransferFlag.fill(1)
        # PretrainedDict = {k: v for k, v in PretrainedDict.items() if k in ModelDict and 'classifier' not in k and 'fc' not in k}
    elif PreTrained == 2 or PreTrained == 4:
        # Get weight if pretrained weight has the partly same structure but diffetnent keys
        # initialize keys and values to keep the original order
        if len(PretrainedDict) == 1:
            PretrainedDict = PretrainedDict['model'] # for convnext pth
        elif len(PretrainedDict) == 4:
            PretrainedDict = PretrainedDict['model_state'] # for SegNet pyth
        
        OldDictKeys = list(PretrainedDict.keys())
        OldValues = list(PretrainedDict.values())
        NewDictKeys = list(ModelDict.keys())
        NewValues = list(ModelDict.values())
        
        Len1 = len(PretrainedDict)
        Len2 = len(ModelDict)
        LenFlag =  Len1 > Len2
        MaxLen = max(Len1, Len2)
        
        Count = IdxNew = IdxOld = 0
        for _ in range(MaxLen):
            if IdxOld >= Len1 or IdxNew >= Len2:
                break
            
            OldKey = OldDictKeys[IdxOld]
            OldVal = OldValues[IdxOld]
            NewKey = NewDictKeys[IdxNew]
            NewVal = NewValues[IdxNew]
            
            TransferFlag = ignoreTransfer(NewKey, IgnoreStrList, TransferFlag, IdxNew)
            
            if not (PreTrained == 4 or DistMode):
                Flag = 0
                if 'classifier.' in OldKey.lower():
                    PretrainedDict.pop(OldKey)
                    IdxOld += 1
                    Flag = 1
                if 'classifier.' in NewKey.lower():
                    IdxNew += 1
                    Flag = 1
                if Flag:
                    continue # for multi classifier
                    # break
            
            SEJudge = False # "selayer" in NewKey.lower() or "mclayer" in NewKey.lower()
            if OldVal.shape == NewVal.shape:
                if not (SkipSELayer and SEJudge): # comment this for other projects
                    PretrainedDict[NewKey] = PretrainedDict.pop(OldKey)
                    TransferFlag = ignoreTransfer(NewKey, IgnoreStrList, TransferFlag, IdxNew, 1)
                    Count += 1
            elif LenFlag:
                if SEJudge:
                    IdxOld -= 1
                else:
                    IdxNew -= 1
                # PretrainedDict.pop(OldKey)
            else:
                IdxOld -= 1
            IdxNew += 1
            IdxOld += 1

            if DropLast:
                if LenFlag and IdxOld == len(OldDictKeys) - 2:
                    break
                elif IdxNew == len(NewDictKeys) - 2:
                    break
    
        print('The number of transferred layers: %d' %(Count))
 
    ModelDict.update(PretrainedDict)
    Model.load_state_dict(ModelDict, strict=False)
    
    Model = wightFrozen(Model, FreezeWeight, TransferFlag, PreTrained)

    return Model


def optimizerChoice(NetParam, lr, Choice='Adam', **kwargs: Any):
    
    CallDict = {
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'adamax': optim.Adamax,
    'sparseadam': optim.SparseAdam,
    'sgd': optim.SGD,
    'asgd': optim.ASGD,
    'rmsprop': optim.RMSprop,
    'rprop': optim.Rprop,
    'lbfgs': optim.LBFGS,
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad,
    }
    
    Optimizer = CallDict[Choice](NetParam, lr=lr, **kwargs)
        
    return Optimizer


def readImagePil(ImgPath):
    try:
        Img = Image.open(ImgPath).convert("RGB")
    except:
        Img = None
    return Img


def readMaskPil(MaskPath):
    try:
        Mask = Image.open(MaskPath)
        if Mask.mode != "L" and Mask.mode != "P":
            print("Mask mode should be L or P. Got: {}".format(Mask.mode))
        return Mask
    except:
        return None


def reduceTensor(InpTensor: torch.Tensor) -> torch.Tensor:
    Size = dist.get_world_size() if dist.is_initialized() else 1
    InpTensorClone = InpTensor.detach().clone()
    # dist_barrier()
    dist.all_reduce(InpTensorClone, op=dist.ReduceOp.SUM)
    InpTensorClone /= Size
    return InpTensorClone


def tensor2PythonFloat(
    InpTensor: Union[int, float, torch.Tensor], IsDistributed: bool
) -> Union[int, float, np.ndarray]:
    if IsDistributed and isinstance(InpTensor, torch.Tensor):
        InpTensor = reduceTensor(InpTensor=InpTensor)

    if isinstance(InpTensor, torch.Tensor) and InpTensor.numel() > 1:
        # For IOU, we get a C-dimensional tensor (C - number of classes)
        # so, we convert here to a numpy array
        return InpTensor.cpu().numpy()
    elif hasattr(InpTensor, "item"):
        return InpTensor.item()
    elif isinstance(InpTensor, (int, float)):
        return InpTensor * 1.0
    else:
        raise NotImplementedError(
            "The data type is not supported yet in tensor_to_python_float function"
        )
    

class Colormap(object):
    """
    Generate colormap for visualizing segmentation masks or bounding boxes.

    This is based on the MATLab code in the PASCAL VOC repository:
        http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """

    def __init__(self, n: Optional[int] = 256, normalized: Optional[bool] = False):
        super(Colormap, self).__init__()
        self.n = n
        self.normalized = normalized

    @staticmethod
    def getBitAtIdx(val, idx):
        return (val & (1 << idx)) != 0

    def getColourMap(self) -> np.ndarray:
        dtype = "float32" if self.normalized else "uint8"
        color_map = np.zeros((self.n, 3), dtype=dtype)
        for i in range(self.n):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (self.getBitAtIdx(c, 0) << 7 - j)
                g = g | (self.getBitAtIdx(c, 1) << 7 - j)
                b = b | (self.getBitAtIdx(c, 2) << 7 - j)
                c = c >> 3

            color_map[i] = np.array([r, g, b])
        color_map = color_map / 255 if self.normalized else color_map
        return color_map

    def getBoxColourCodes(self) -> List:
        box_codes = []

        for i in range(self.n):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (self.getBitAtIdx(c, 0) << 7 - j)
                g = g | (self.getBitAtIdx(c, 1) << 7 - j)
                b = b | (self.getBitAtIdx(c, 2) << 7 - j)
                c = c >> 3
            box_codes.append((int(r), int(g), int(b)))
        return box_codes

    def getColorMapList(self) -> List:
        cmap = self.getColourMap()
        cmap = np.asarray(cmap).flatten()
        return list(cmap)


def saveModel(SavePath, Model):
  if isinstance(Model, torch.nn.DataParallel):
    StateDict = Model.module.state_dict()
  else:
    StateDict = Model.state_dict()
  torch.save(StateDict, SavePath)
  
  
def listPrinter(ListName, ListValues, LineNum=5, **kwargs: Any):
    Len1 = len(ListName)
    Len2 = len(ListValues)
    assert Len1 == Len2, "Got not consistent len %d and %d" % (Len1, Len2)
    
    LineNum = 5
    Remainder = Len1 % LineNum
    if Remainder > 0 and Remainder <3:
        LineNum = 4
        Remainder = Len1 % LineNum
        if Remainder > 0 and Remainder < 2:
            LineNum = 6
            Remainder = Len1 % 6
            if Remainder > 0 and Remainder < 3:
                LineNum = 4
                  
    PrintStr = ''
    for Idx in range(Len1):
        if Idx > 0 and Idx % LineNum == 0:
            PrintStr += '\n'
        
        Value = ListValues[Idx]
        if isinstance(Value, float):
            if Value > 1:
                Value = round(Value, 2)
            else:
                Value = round(Value, 5)
        
        PrintStr += '%s: %s, ' % (ListName[Idx], colorText(str(Value), **kwargs))
        # %r for bool printing
    
    print(PrintStr)
    
    
def callMethod(self, ElementName):
    return getattr(self, ElementName)


def setMethod(self, ElementName, ElementValue):
    return setattr(self, ElementName, ElementValue)


def indicesSameEle(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


def getOnlyFolderNames(RootPath):
    FolderNames = [Name for Name in os.listdir(RootPath) \
        if os.path.isdir(os.path.join(RootPath, Name))]
    return FolderNames


def getMetaLogPath(MetaRootPath):
    Metadata = None
    for File in os.listdir(MetaRootPath):
        if File.endswith(".csv") and 'log_' in File:
            Metadata = MetaRootPath + '/' + File # /log_rotation_shuffle.csv
            print('Metadata is %s' % Metadata)
            break
    return Metadata


def getBestModelPath(FolderPath):
    if os.path.isdir(FolderPath):
        SaveModels = os.listdir(FolderPath)
        if len(SaveModels) < 1:
            return ''
        Idx = [re.findall(r'%s(\d+)' % 'epoch', SaveModels[i])[0] for i in range(len(SaveModels))]
        # Idx.sort()
        Idx = list(map(int, Idx))
        BestIdx = Idx.index(max(Idx))
        IndexTemp = BestIdx
        SaveModel = SaveModels[BestIdx]
        while 'minloss' not in SaveModel:
            if IndexTemp == 0:
                IndexTemp = BestIdx
                SaveModel = SaveModels[BestIdx]
                while 'maxacc' not in SaveModel:
                    if IndexTemp == 0:
                        SaveModel = SaveModels[-1]
                        break
                    else:
                        IndexTemp -= 1
                        SaveModel = SaveModels[BestIdx]
                break
            else:
                IndexTemp -= 1
                SaveModel = SaveModels[BestIdx]
                
        return FolderPath + SaveModel
    else:
        return ''


def getWeightName(MetaRow: Dict, TargetExp):
    SupMethod = MetaRow['Supervision']
    ModelName = MetaRow['Model']

    Epochs = MetaRow['NumEpochs']
    BatchSize = MetaRow['BatchSize']
    Loss = MetaRow['Loss']
    MixMethod = MetaRow['MixMethod']
    MixMinCropRatio = MetaRow['MixMinCropRatio']
    MixMethodStr = ('_%s%s' % (MixMethod, MixMinCropRatio.replace("0.", ""))) if MixMethod else ''
    
    # general string
    SaveStr = ''
    if 'common' not in SupMethod:
        NumViews = MetaRow.get('NumViews', '')
        SaveStr += 'v%s_' % NumViews if NumViews else ''
    
        BatchShuffle = MetaRow.get('BatchShuffle', '')
        AugLikeClr = MetaRow.get('AugLikeClr', '')
        SaveStr += 'bshuffle_' if 'true' in BatchShuffle.lower() else ''
        SaveStr += 'clraug_' if 'true' in AugLikeClr.lower() else ''
        
        CropMode = MetaRow.get('CropMode', '')
        CropRatio = MetaRow.get('CropRatio', '')
        ShiftMode = MetaRow.get('ShiftMode', '')
        MinCropRatio = MetaRow.get('MinCropRatio', '')
        RandomPos = MetaRow.get('RandomPos', '')
        CentreRand = MetaRow.get('CentreRand', '')
        SelfsupValid = MetaRow.get('SelfsupValid', '')
        
        if CropMode:
            if CropMode != str(2):
                SaveStr += 'cr%s_' % CropMode
        
        SaveStr += ('r_' + CropRatio.replace("0.", "")) if CropRatio else ''
        SaveStr += ('minr' + MinCropRatio.replace("0.", "") + '_') if MinCropRatio else ''
        SaveStr += 'swin_' if 'true' in ShiftMode.lower() else ''
        SaveStr += 'centre_' if 'true' not in RandomPos.lower() else ''
        SaveStr += 'crand_' if 'true' in CentreRand.lower() else ''
        SaveStr += 'valid_' if 'true' in SelfsupValid.lower() else ''

    IdenStr = ''
    if 'rotation' in SupMethod:
        ## rotation
        RotDegree = MetaRow['RotDegree']
        RegionRot = MetaRow['RegionRot']
        AngleShaking = MetaRow['AngleShaking']
        
        IdenStr += RotDegree if RotDegree else ''
        IdenStr += 'rr_' if 'true' in RegionRot.lower() else ''

        SaveStr += 'shake_' if 'true' in AngleShaking.lower() else ''
    elif 'position' in SupMethod:
        ## position
        NumSlices = MetaRow['NumSlices'] 
        SliceGap = MetaRow['SliceGap']
        
        IdenStr += 's%sg%s_' % (NumSlices, SliceGap)
        
    SaveStr += 'e' + Epochs
    SaveStr += 'bs%s%s' % (BatchSize, '_')
    SaveStr += (Loss + '_') if Loss != 'cross_entropy' else ''
    # SaveStr += '_cw' if 'false' not in ClassWeight.lower() else ''
    # SaveStr += '_lb' + LabelSmooth if '0' not in LabelSmooth else ''
    
    SaveStr += 'Exp%s' % TargetExp
    
    SaveName = '%s_%s%s_%s%s' % (SupMethod, ModelName, MixMethodStr, IdenStr, SaveStr)
    
    return SaveName


def pointCentreRotation(Centre, Point, Angle):
    # Rotate a point clockwise by a given Angle around a given origin.
    Angle = math.radians(Angle)
    # [x, y, w, h] format
    Ox, Oy = Centre
    Px, Py = Point
        
    Qx = Ox + math.cos(Angle) * (Px - Ox) - math.sin(Angle) * (Py - Oy)
    Qy = Oy + math.sin(Angle) * (Px - Ox) + math.cos(Angle) * (Py - Oy)
    
    return [round(Qx, 3), round(Qy, 3)]


def boxPointRotation(Box, Angle):
    '''MaskBox = boxPointRotation(MaskBox, RotAngle2)'''
    # [Left, Upper, Right, Lower]
    Cx = (Box[2] - Box[0]) / 2
    Cy = (Box[3] - Box[1]) / 2
    Centre = [Cx, Cy]
    
    UpperLeft = pointCentreRotation(Centre, [Box[0], Box[1]], Angle)
    UpperRight = pointCentreRotation(Centre, [Box[2], Box[1]], Angle)
    LowerLeft = pointCentreRotation(Centre, [Box[0], Box[3]], Angle)
    LowerRight = pointCentreRotation(Centre, [Box[2], Box[3]], Angle)
    
    XList= [UpperLeft[0], UpperRight[0], LowerLeft[0], LowerRight[0]]
    YList= [UpperLeft[1], UpperRight[1], LowerLeft[1], LowerRight[1]]
    XList.sort()
    YList.sort()
    
    return [XList[0], YList[0], XList[-1], YList[-1]]


def getFilePathsFromSubFolders(WalkPath):
    return [os.path.join(Root, File) \
        for Root, Dirs, Files in os.walk(WalkPath) for File in Files]
   
    
def getLoaderBatch(opt):
    if 'common' in opt.sup_method or 'shuffle' in opt.sup_method:
        LoaderBatch = opt.batch_size
    else:
        LoaderBatch = opt.batch_size / opt.views
        # self supervised dataloader provide different number of data
        if LoaderBatch.is_integer():
            # keep real batch size
            LoaderBatch = round(LoaderBatch)
        else:
            LoaderBatch = makeDivisible(LoaderBatch, 4)
    return LoaderBatch


def deviceInit(opt):
    DataParallel = False
    CUDA_AVAI = torch.cuda.is_available()
    if CUDA_AVAI:
        if len(opt.gpus) > 1:
            DeviceStr = 'cuda'
            DataParallel = True
        else:
            DeviceStr = 'cuda:' + str(opt.gpus[0])
    else:
        DeviceStr = 'cpu'
    return DataParallel, DeviceStr
