## Import module
 # path manager
import os
 # data processing
import time
from datetime import datetime
 # torch module
import torch
 # my module
from opts import Opts
from lib.trainer import Trainer
from lib.params_init import paramsInit
from lib.utils import (listPrinter, seedSetting, 
                       deviceInit, expFolderCreator, writeCsv)


class MainTrain(object):
    def __init__(self, opt) -> None:
        seedSetting(RPMode=False)
        
        self.paramsInit(opt)
        self.filePathInit(opt)
        self.logFieldInit(opt)
        
        self.opt = opt
        
    def paramsInit(self, opt):
        # device
        self.DataParallel, self.DeviceStr = deviceInit(opt)
        opt.device = torch.device(self.DeviceStr)
    
    def filePathInit(self, opt):
        if opt.num_supplement == 0:
            self.TargetExp = None
            self.Sup = False
        else:
            self.Sup = True
            
        ExpType = opt.task
        if '5nn' in opt.clsval_mode:
            ExpType += '_' + '5nn'
        elif 'common' not in opt.sup_method:
            ExpType += '_' + opt.sup_method
            
        DestPath, self.ExpCount = expFolderCreator(ExpType=ExpType, ExpLevel=opt.exp_level, TargetExp=self.TargetExp)
        
        self.ExpLogPath = './exp/%s/%s/log_%s.csv' % (ExpType, opt.exp_level, opt.sup_method)
        self.InputLogPath = DestPath + '/input_param.csv'
        opt.dest_path = DestPath
        
    def logFieldInit(self, opt):
        LogField = [
            'Task', 'Supervision', 'Dataset', 'NumClasses', 
            'BatchSize', 'Resolution', 'MeanStd',
            'Model', 'Optimizer', 'Schedular', 'WeightDecay', 
            'Loss', 'MetricName', 'MetricType', 'CollateFn', 
        ] # Define header
        LogInfo = [
            opt.task, opt.sup_method, opt.setname, opt.num_classes, 
            opt.batch_size, opt.resize_res, opt.use_meanstd,
            opt.model_name, opt.optim, opt.schedular, opt.weight_decay, 
            opt.loss_name, opt.metric_name, opt.metric_type, opt.collate_fn_name, 
        ]
        listPrinter(['Device'] + LogField, [self.DeviceStr] + LogInfo)
        
        FirstLogField = ['exp', 'date']
        EndLogField = [
            'LrDecay', 'PreTrained', 'FreezeWeight', 'NumEpochs', 
            'NumberofSplit', 'NumRepeat', 'Sup', 'LrRate', 
        ]
        FirstLogInfo = [self.ExpCount, datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        EndLogInfo = [
            opt.lr_decay, opt.pretrained, opt.freeze_weight, opt.epochs,
            opt.num_split, opt.num_repeat, self.Sup, opt.lr, 
        ]
        self.LogField = FirstLogField + LogField + EndLogField
        self.LogInfo = FirstLogInfo + LogInfo + EndLogInfo
        
        if 'common' in opt.sup_method:
            WeightName = opt.weight_name
            if WeightName:
                WeightName = os.path.splitext(os.path.basename(WeightName))[0] # get weight name
                # remove model name from weight name
                WeightName = WeightName.replace(opt.model_name.lower() + '_', '')
            self.LogField.extend(['WeightName'])
            self.LogInfo.extend([WeightName])
        else:
            # if opt.mix_method is not None:
            ListName = ['MixMethod', 'MixupAlpha', 'MixupGamma', 'MixupKappa', 'MixupBias', 
                        'MixMinCropRatio', 'MixMaskRot']
            ListValues = [opt.mix_method, opt.mix_alpha, opt.mix_gamma, opt.mix_kappa, opt.mix_base_bias, 
                        opt.mix_mincrop_ratio, opt.mix_maskrot]

            self.LogField.extend(ListName)
            self.LogInfo.extend(ListValues) 
            if opt.mix_method is not None:
                listPrinter(ListName, ListValues)
            
        ListName = []
        ListValues = []
        
        if 'classification' in opt.task:
            if 'common' not in opt.sup_method:
                ListName.extend(['SelfsupValid', 'NumViews'])
                ListValues.extend([opt.selfsup_valid, opt.views])
                
                ListName.extend(['BatchShuffle', 'CropMode', 'CropRatio', 'MinCropRatio', 
                                 'RandomPos', 'CentreRand', 'CropResize', 'RmBackground'])
                ListValues.extend([opt.batch_shuffle, opt.crop_mode, opt.crop_ratio, opt.mincrop_ratio, 
                                   opt.random_pos, opt.centre_rand, opt.crop_resize, opt.remove_background])
                
                if 'rot' in opt.sup_method:
                    ListName.extend(['RotDegree', 'RegionRot'])
                    ListValues.extend([opt.rot_degree, opt.region_rot])

            listPrinter(ListName, ListValues)
            self.LogField.extend(ListName)
            self.LogInfo.extend(ListValues)

        elif 'segmentation' in opt.task:
            ListName.extend(['SegModel', 'SegHead', 'UseSepConv'])
            ListValues.extend([opt.seg_model_name, opt.seg_head_name, opt.use_sep_conv])
            
            listPrinter(ListName, ListValues)
            self.LogField.extend(ListName)
            self.LogInfo.extend(ListValues)
            
        writeCsv(self.InputLogPath, self.LogField, self.LogInfo)
       
    def training(self):
        if self.opt.num_split == 1 and self.opt.num_repeat > self.opt.num_split:
            SplitLoop = [0] * self.opt.num_repeat
        else:
            SplitLoop = range(self.opt.num_split)

        for i, Split in enumerate(SplitLoop):
            StopSign = Split + self.opt.num_supplement
            
            self.MyTrainer = Trainer(self.opt, self.DataParallel) # training class
            self.MyTrainer.run(i, Split)
            
            ## Writing results
            self.MyTrainer.writeRunningMetrics()
            if self.MyTrainer.ValDL:
                self.MyTrainer.writeMetricsRecord()
                self.MyTrainer.writeBestMetrics()
            if i >= self.opt.num_repeat - 1 or StopSign >= self.opt.num_repeat - 1:
                if self.MyTrainer.ValDL:
                    self.MyTrainer.writeAvgBestMetrics()
                break
        
    def writeLogFile(self, TimeCost):
        # Write input and output param in log file
        self.LogInfo[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.LogInfo.append(TimeCost)
        NewFieldNames = ['TimeCost']
        if self.MyTrainer.ValDL:
            if 'classification' in self.opt.task:
                self.LogInfo.extend(self.MyTrainer.AvgBestMetric[1:])
                NewFieldNames.append('Accuracy')
                if opt.sup_metrics:
                    NewFieldNames.extend(['Recall', 'Specificity', 'Precision', 'F1Score'])
            elif 'segmentation' in opt.task:
                self.LogInfo.extend(self.MyTrainer.AvgBestMetric[1:])
                NewFieldNames.extend(['mIoU'])
        else:
            self.LogInfo.append(self.MyTrainer.BestLoss)
            NewFieldNames.append('Train loss')
            if 'classification' in self.opt.task:
                self.LogInfo.append(self.MyTrainer.BestComMetric)
                NewFieldNames.append('Train accuracy')
            elif 'segmentation' in self.opt.task:
                self.LogInfo.append(self.MyTrainer.BestComMetric)
                NewFieldNames.append('Train mIoU')

        writeCsv(self.ExpLogPath, self.LogField, self.LogInfo, NewFieldNames)


if __name__ == "__main__":
    Tick0 = time.perf_counter()
    
    opt = Opts().parse()
    opt = paramsInit(opt)
    MyTrain = MainTrain(opt)

    MyTrain.training()
    
    TimeCost = time.perf_counter() - Tick0
    print('Finish training using: %.4f minutes' % (TimeCost / 60))
    
    MyTrain.writeLogFile(TimeCost)