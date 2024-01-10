## Import module
 # path manager
import os
from pathlib import Path
 # data processing
import csv
import math
import numpy as np
import pandas as pd
 # torch module
from torch.utils.data import DataLoader
 # my module
from .utils import getSetPath
from .statistics import Statistics
from ..loss_fn import getLossFn
from ..data import getImgPath, getMyDataset, buildCollateFn
from ..utils import (writeCsv, saveModel, 
                     listPrinter, workerManager,
                     getLoaderBatch)


class BaseTrainer(object):
    def __init__(self, opt, DataParallel) -> None:
        opt = self.paramsInit(opt)
        self.filePathInit(opt)
        self.TrainState = Statistics(opt, State='train')
        self.ValState = Statistics(opt, State='val')
        
        self.opt = opt
        
        self.DataParallel = DataParallel
        
        LossFn = getLossFn(opt)
        self.LossFn = LossFn.to(opt.device)
        
    def paramsInit(self, opt):
        self.Split = None
        self.Repeat = 0 # for best model saver
        
        self.TrainDL = None
        self.ValDL = None
        
        self.LrList = None
        self.CurrentLrRate = None
        
        self.PinMemory, self.NumWorkers = workerManager(opt.batch_size, opt.num_workers)
        
        self.TrainSet, self.TestSet = getImgPath(opt.dataset_path, opt.num_split, Mode=opt.get_path_mode)
        return opt
         
    def splitInit(self, opt):
        self.BestEpoch = 0
        self.BestLoss = 9e5
        self.BestComMetric = 0
        self.LastMinLossPath = ''
        self.LastMetricPath = ''
        self.BestStopIndicator = self.BestLoss if opt.metric_name == 'loss' \
            else self.BestComMetric # Best metric indicator
    
    def filePathInit(self, opt):
        if opt.dest_path is not None:
            self.ModelSavePath =  opt.dest_path + '/model'  # Save weight
            self.TrainRecordPath = opt.dest_path + '/metrics/record'  # Save indicators during training
            self.TrainMetricsPath =  opt.dest_path + '/best_metrics.csv'  # Save metrics
            self.TrainRecordPathSingle = opt.dest_path + '/metrics/'
            Path(self.TrainRecordPathSingle).mkdir(parents=True, exist_ok=True)
    
    def dataLoaderSetting(self, opt):
        TrainSet, TestSet = getSetPath(self.TrainSet, self.TestSet, self.Split)
        TargetTrainSet = TargetTestSet = None
        
        opt.split = self.Split # for mean std init         
        self.TestImgs = None
        if 'common' in opt.sup_method:
            self.TrainImgs = getMyDataset(opt, TrainSet, TargetTrainSet, IsTraining=True)
            self.TestImgs = getMyDataset(opt, TestSet, TargetTestSet, IsTraining=False)
            if '5nn' in self.opt.clsval_mode:
                self.MemImgs = getMyDataset(opt, TrainSet, TargetTrainSet, IsTraining=False)
        else:
            if not opt.selfsup_valid:
                SetPath = TrainSet # self.TrainSet[0] 
                if opt.extend_val:
                    SetPath.extend(TestSet)
                    if TargetTrainSet is not None:
                        TargetTrainSet.extend(TargetTestSet)
                self.TrainImgs = getMyDataset(opt, SetPath, TargetTrainSet, IsTraining=True)
            else:
                self.TrainImgs = getMyDataset(opt, TrainSet, TargetTrainSet, IsTraining=True)
                self.TestImgs = getMyDataset(opt, TestSet, TargetTestSet, IsTraining=False)
    
        LoaderBatch = getLoaderBatch(opt)
        
        CollateFn = buildCollateFn(opt)
        if 'common' in opt.sup_method or opt.selfsup_valid:
            self.TrainDL = DataLoader(self.TrainImgs, LoaderBatch, num_workers=self.NumWorkers, shuffle=opt.loader_shuffle, 
                                      pin_memory=self.PinMemory, collate_fn=CollateFn, drop_last=opt.drop_last)
            self.ValDL = DataLoader(self.TestImgs, LoaderBatch, num_workers=self.NumWorkers, shuffle=opt.loader_shuffle, 
                                    pin_memory=self.PinMemory, collate_fn=CollateFn, drop_last=opt.drop_last)
            if '5nn' in opt.clsval_mode:
                self.MemDL = DataLoader(self.MemImgs, LoaderBatch, num_workers=self.NumWorkers, shuffle=False, 
                                      pin_memory=self.PinMemory, collate_fn=CollateFn, drop_last=False)
        else:
            self.TrainDL = DataLoader(self.TrainImgs, LoaderBatch, num_workers=self.NumWorkers, shuffle=opt.loader_shuffle, 
                                    pin_memory=self.PinMemory, collate_fn=CollateFn, drop_last=opt.drop_last)
            
    def modelSaver(self, opt, Epoch: int, Model):
        if self.ValDL:
            Loss = self.ValState.Loss
            ComMetric = self.ValState.ComMetric
        else:
            Loss = self.TrainState.Loss
            ComMetric = self.TrainState.ComMetric
        
        SaveModelStr = ''
        SaveModelFlag = 0
        if Loss < self.BestLoss:
            SaveModelFlag = 2
            self.BestLoss = Loss
            SaveModelStr += '_minloss'
            if 'loss' in opt.metric_name:
                self.BestStopIndicator = self.BestLoss
                self.BestEpoch = Epoch
        if ComMetric > self.BestComMetric:
            SaveModelFlag += 3
            self.BestComMetric = ComMetric
            if 'classification' in opt.task:
                SaveModelStr += '_maxacc'
            elif 'segmentation' in opt.task:
                SaveModelStr += '_maxiou'
            if 'loss' not in opt.metric_name:
                self.BestStopIndicator = self.BestComMetric
                self.BestEpoch = Epoch

        if Epoch in opt.save_point:
            SaveModelFlag = 1
        # # elif Epoch == self.Epochs:
        # #     SaveModelFlag = 5
        
        if SaveModelFlag == 1 or (SaveModelFlag > 0 and Epoch >= opt.milestones):
        #     # if Epoch > opt.save_point[-1]:
            SaveModelStr = '%s_epoch%d%s%s_f%s.pth' \
                % (opt.model_name, Epoch, SaveModelStr, 
                    ('_valrpt%d' % (self.SaveSign)) if 'common' in opt.sup_method else '', SaveModelFlag)
            SavePath = '%s/%s' % (self.ModelSavePath, SaveModelStr)
            saveModel(SavePath, Model)
            
            if SaveModelFlag == 2:
                if os.path.exists(self.LastMinLossPath):
                    if '_f2' in self.LastMinLossPath or \
                        ('loss' in opt.metric_name and '_f5' in self.LastMinLossPath):
                        os.remove(self.LastMinLossPath)
                        # print("The file has been deleted successfully")
                self.LastMinLossPath = SavePath
            elif SaveModelFlag == 3:
                if os.path.exists(self.LastMetricPath):
                    if '_f3' in self.LastMetricPath or \
                        ('loss' not in opt.metric_name and '_f5' in self.LastMetricPath):
                        os.remove(self.LastMetricPath)
                        # print("The file has been deleted successfully")
                self.LastMetricPath = SavePath
            elif SaveModelFlag == 5:
                if os.path.exists(self.LastMinLossPath):
                    os.remove(self.LastMinLossPath)
                if os.path.exists(self.LastMetricPath):
                    os.remove(self.LastMetricPath)
                self.LastMinLossPath = SavePath
                self.LastMetricPath = SavePath

    def bestManager(self, opt, Epoch: int, Model):
        Epoch = Epoch + 1 # for number convenience
        self.SaveSign = max(self.Split, self.Repeat) + 1
        self.modelSaver(opt, Epoch, Model)
        
        if 'classification' in opt.task: 
            if self.ValDL:
                print('Epoch: [%d/%d] \tCrossValid: [%d/%d], \tLearning rate: %.6f' \
                    % (Epoch, opt.epochs, self.Split + 1, opt.num_split, self.CurrentLrRate))
                
                ListName = ['Best', 'BEpoch', 'Accuracy', 'ValAcc', 'Loss', 'ValLoss']
                ListValue = [self.BestStopIndicator, self.BestEpoch, self.TrainState.Top1Accuracy, 
                             self.ValState.Top1Accuracy, self.TrainState.Loss, self.ValState.Loss]
            
                if self.opt.sup_metrics:
                    ListName.extend(['AvgRecall', 'AvgPrecis', 'AvgF1Score', 'AvgSpec'])
                    ListValue.extend([self.ValState.AvgRecall, self.ValState.AvgPrecision, 
                                      self.ValState.AvgF1Score, self.ValState.AvgSpecificity]) #  time.perf_counter() - Tick1
                else:
                    ListName.extend(['Top5Acc', 'ValTop5Acc'])
                    ListValue.extend([self.TrainState.Top5Accuracy, self.ValState.Top5Accuracy])
                    
                listPrinter(ListName, ListValue, Mode=2)
            else:
                print('Epoch: [%d/%d], \tLearning rate: %.6f' % (Epoch, opt.epochs, self.CurrentLrRate))
                
                ListName = ['Best', 'BEpoch', 'Loss']
                ListValue = [self.BestStopIndicator, self.BestEpoch, self.TrainState.Loss]
                ListName.append('Accuracy')
                ListValue.append(self.TrainState.Top1Accuracy)
                
                if not self.opt.sup_metrics:
                    ListName.append('Top5Accuracy')
                    ListValue.append(self.TrainState.Top5Accuracy)
                
                listPrinter(ListName, ListValue, Mode=2)
                
        elif 'segmentation' in opt.task:
            if self.ValDL:
                print('Epoch: [%d/%d] \tCrossValid: [%d/%d], \tLearning rate: %.6f' \
                    % (Epoch, opt.epochs, self.Split + 1, opt.num_split, self.CurrentLrRate))
                
                ListName = ['Best', 'BEpoch', 'mIoU', 'ValmIoU', 'Loss', 'ValLoss']
                ListValue = [self.BestStopIndicator, self.BestEpoch, self.TrainState.mIoU, self.ValState.mIoU, 
                             self.TrainState.Loss, self.ValState.Loss] #  time.perf_counter() - Tick1
                listPrinter(ListName, ListValue, Mode=2)
            else:
                print('Epoch: [%d/%d], \tLearning rate: %.6f' % (Epoch, opt.epochs, self.CurrentLrRate))
                
                ListName = ['Best', 'BEpoch', 'mIoU']
                ListValue = [self.BestStopIndicator, self.BestEpoch, self.TrainState.mIoU]
                    
                ListName.append('Loss')
                ListValue.append(self.TrainState.Loss)
                    
                listPrinter(ListName, ListValue, Mode=2)
            
    def writeRunningMetrics(self):
        # Write avrage training metrics record
        OutputFieldNames = ['LrRate', 'BatchSize', 'ResizeResolution', 'TrainLoss']
        OutputExcel = {'LrRate': self.LrList, 'BatchSize': self.opt.batch_size, 
                       'ResizeResolution': self.opt.resize_res, 'TrainLoss': self.TrainState.LossList}
        
        if 'classification' in self.opt.task:
            OutputFieldNames.extend(['TrainAccracy'])
            OutputExcelCls = {'TrainAccracy': self.TrainState.Top1AccuracyList}
            OutputExcel.update(OutputExcelCls)
            
            if not self.opt.sup_metrics:
                OutputFieldNames.extend(['TrainTop5Accracy'])
                OutputExcelCls = {'TrainTop5Accracy': self.TrainState.Top5AccuracyList}
                OutputExcel.update(OutputExcelCls)
            
            if self.ValDL:
                OutputFieldNames.extend(['ValLoss', 'ValAccracy'])
                OutputExcelTest = {'ValLoss': self.ValState.LossList, 'ValAccracy': self.ValState.Top1AccuracyList}
                OutputExcel.update(OutputExcelTest)
                if self.opt.sup_metrics:
                    OutputFieldNames.extend(['AvgRecall', 'AvgPrecision', 'AvgF1Score', 'AvgSpecificity'])
                    OutputExcelTest = {'AvgRecall': self.ValState.AvgRecallList, 'AvgPrecision': self.ValState.AvgPrecisionList,
                                        'AvgF1Score': self.ValState.AvgF1ScoreList, 'AvgSpecificity': self.ValState.AvgSpecificityList}
                    OutputExcel.update(OutputExcelTest)
                else:
                    OutputFieldNames.extend(['ValTop5Accracy'])
                    OutputExcelCls = {'ValTop5Accracy': self.ValState.Top5AccuracyList}
                    OutputExcel.update(OutputExcelCls)
                    
        elif 'segmentation' in self.opt.task:
            OutputFieldNames.extend(['TrainIoU'])
            OutputExcelSeg = {'TrainIoU': self.TrainState.mIoUList}
            OutputExcel.update(OutputExcelSeg)
            OutputFieldNames.extend(['ValLoss', 'ValIoU'])
            OutputExcelSeg = {'ValLoss': self.ValState.LossList, 'ValIoU': self.ValState.mIoUList}
            OutputExcel.update(OutputExcelSeg)

        Output = pd.DataFrame(OutputExcel)
        Output.to_csv(self.TrainRecordPath + '_valrpt{}.csv'.format(self.SaveSign), 
                      columns=OutputFieldNames, encoding='utf-8')
        
    def writeMetricsRecord(self):
        # Write training record for each class
        if self.ValDL:
            if 'classification' in self.opt.task and self.opt.sup_metrics:
                    self.ValState.RecallDF.to_csv(self.TrainRecordPathSingle + 'recall_valrpt{}.csv'.format(self.SaveSign), 
                                        index=False, encoding='utf-8')
                    self.ValState.PrecisionDF.to_csv(self.TrainRecordPathSingle + 'precision_valrpt{}.csv'.format(self.SaveSign), 
                                            index=False, encoding='utf-8')
                    self.ValState.F1ScoreDF.to_csv(self.TrainRecordPathSingle + 'f1score_valrpt{}.csv'.format(self.SaveSign),
                                        index=False, encoding='utf-8')
                    self.ValState.SpecificityDF.to_csv(self.TrainRecordPathSingle + 'specificity_valrpt{}.csv'.format(self.SaveSign), 
                                            index=False, encoding='utf-8')
                
            elif 'segmentation' in self.opt.task:
                self.ValState.IoUDF.to_csv(self.TrainRecordPathSingle + 'iou_valrpt{}.csv'.format(self.SaveSign), 
                                    index=False, encoding='utf-8')

    def writeBestMetrics(self):
        # Export and write the best result
        if self.ValDL:
            if self.opt.metric_name == 'loss':
                Idx = self.ValState.LossList.index(min(self.ValState.LossList))
            elif self.opt.metric_name == 'iou':
                Idx = self.ValState.mIoUList.index(max(self.ValState.mIoUList))
            elif self.opt.metric_name == 'ap':
                Idx = self.ValState.mAPList.index(max(self.ValState.mAPList))
            else: # accuracy
                Idx = self.ValState.Top1AccuracyList.index(max(self.ValState.Top1AccuracyList))
                
            if 'classification' in self.opt.task:
                BestAccuracy = self.ValState.Top1AccuracyList[Idx]
                DfBest = ['valrpt_{}'.format(self.SaveSign), BestAccuracy]
                self.MetricsFieldNames = ['K-Fold', 'BestAccuracy'] 
                
                if self.opt.sup_metrics:
                    BestRecall = self.ValState.AvgRecallList[Idx]
                    BestSpecificity = self.ValState.AvgSpecificityList[Idx]
                    BestPrecision = self.ValState.AvgPrecisionList[Idx]
                    BestF1Score = self.ValState.AvgF1ScoreList[Idx]
                    
                    DfBest.extend([BestRecall, BestSpecificity, BestPrecision, BestF1Score])
                    self.MetricsFieldNames.extend(['BestRecall', 'BestSpecificity', 'BestPrecision', 'BestF1Score'])
                    
                writeCsv(self.TrainMetricsPath, self.MetricsFieldNames, DfBest)
                
            elif 'segmentation' in self.opt.task:
                BestmIoU = self.ValState.mIoUList[Idx]
                
                DfBest = ['valrpt_{}'.format(self.Split + 1), BestmIoU]
                self.MetricsFieldNames = ['K-Fold', 'BestmIoU']    
                writeCsv(self.TrainMetricsPath, self.MetricsFieldNames, DfBest)

    def writeAvgBestMetrics(self):
        # Compute and write mean values of metrics in k-fold validation
        MetricsReader = csv.reader(open(self.TrainMetricsPath, 'r'))
        BestMetrics = []
        for Row in MetricsReader:
            BestMetrics.append(Row)
        BestMetrics.pop(0) # remove the header/title in the first row
        RowNum = len(BestMetrics)
        ColNum = len(BestMetrics[0])
        
        Values = np.zeros((ColNum - 1, 1))
        
        Flag = 0
        for i in range(RowNum):
            if 'Average' in BestMetrics[i][0]:
                Flag = 1
                continue
            for j in range(ColNum - 1):
                Values[j] += float(BestMetrics[i][j + 1])
        Values /= RowNum

        self.AvgBestMetric = ['Average'] if Flag == 0 else ['Average_Sup']
        self.AvgBestMetric.extend(list(Values.flatten()))
        writeCsv(self.TrainMetricsPath, self.MetricsFieldNames, self.AvgBestMetric)

    def run(self, Split: int):
        pass
        