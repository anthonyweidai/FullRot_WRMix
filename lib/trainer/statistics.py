import torch
from torch import Tensor

import numpy as np
import pandas as pd

from ..metrics import (topKCorrect, computeClsValMetric, computeIoUBatch)


class Statistics(object):
    def __init__(self, opt, State='train') -> None:
        self.opt = opt
        self.State = State
        
        # Converting iou from [0, 1] to [0, 100]
        self.Scale = 100.0
        
        self.paramsInit()
        
    @torch.inference_mode()    
    def paramsInit(self):
        self.LossList = []
        self.Top1AccuracyList = [] # for classification
        self.Top5AccuracyList = []
        self.ComMetric = 0
        
        if 'classification' in self.opt.task:
            if 'val' in self.State:
                self.RecallDF = pd.DataFrame()
                self.SpecificityDF = pd.DataFrame()
                self.PrecisionDF = pd.DataFrame()
                self.F1ScoreDF = pd.DataFrame()
                
                self.AvgRecallList =  []
                self.AvgPrecisionList = []
                self.AvgF1ScoreList =  []
                self.AvgSpecificityList = []
                
        elif 'segmentation' in self.opt.task:
            self.IoUDF = pd.DataFrame() 
            
            self.mIoUList = []

    
    @torch.inference_mode()          
    def batchInit(self):
        self.NumTotal = 0
        self.RunningLoss = 0
        self.ComMetric = 0 # for save the best model
        
        self.NumTop1Correct = 0 # for classification
        self.NumTop5Correct = 0 # for classification
        
        self.Loss = 0
        
        if 'classification' in self.opt.task:
            self.Top1Accuracy = 0
            self.Top5Accuracy = 0
            if 'val' in self.State:
                self.NumEachClass = np.zeros((self.opt.cls_num_classes, 1), dtype=int) # prevent some class doen't has data
                
                self.AvgRecall = None
                self.AvgPrecision = None
                self.AvgF1Score = None
                self.AvgSpecificity = None
                
                self.TruePositive, self.FalsePositive, self.TrueNegative, self.FalseNegative \
                    = np.zeros((4, self.opt.cls_num_classes), dtype=int)
                                
        elif 'segmentation' in self.opt.task:
            self.UnionCout = np.zeros((self.opt.seg_num_classes, 1), dtype=int)
            self.AreaInter, self.AreaUnion, self.IoU = np.zeros((3, self.opt.seg_num_classes), dtype=float)

    @torch.inference_mode()
    def batchUpdate(self, PredLabel: Tensor, TargetLabel: Tensor, Loss: Tensor, Epoch: int):
        if isinstance(PredLabel, Tensor):
            BatchSize = PredLabel.shape[0] # Could be unequal to the default batchsize
        elif isinstance(PredLabel, dict):
            # Could be unequal to the default batchsize,
            # simsiam takes 2 view of same image as one pair in the batch
            BatchSize = list(PredLabel.values())[0].shape[0] 
        elif isinstance(PredLabel, list):
            # for barlow
            BatchSize = PredLabel[0].shape[0] 
            
        self.NumTotal += BatchSize
        self.RunningLoss += Loss.item() if Loss is not None else 0
        
        if 'classification' in self.opt.task:
            if 'linear' in self.opt.clsval_mode or 'train' in self.State:
                AccuracyComb = topKCorrect(PredLabel, TargetLabel, Topk=self.opt.topk)
                if self.opt.sup_metrics:
                    self.NumTop1Correct += AccuracyComb[0]
                else:
                    NumTop1Correct, NumTop5Correct = AccuracyComb
                    self.NumTop1Correct += NumTop1Correct
                    self.NumTop5Correct += NumTop5Correct
            else:
                self.NumTop1Correct += (PredLabel[:, 0] == TargetLabel).float().sum().item()
            
            if 'val' in self.State:
                if self.opt.sup_metrics:
                    PredLabel = torch.argmax(PredLabel, dim=1)
                    Unique, Counts = np.unique(TargetLabel.cpu().numpy(), return_counts=True)
                    ClassCount = dict(zip(Unique, Counts))
                    for i in range(self.opt.cls_num_classes):
                        if i in list(ClassCount.keys()):
                            self.NumEachClass[i] += ClassCount[i]
                    
                    # Get true/false and positive/negative samples, by superposiiton
                    for i in range(BatchSize):
                        for k in range(self.opt.cls_num_classes):
                            if TargetLabel[i].item() == k:
                                if PredLabel[i] == TargetLabel[i]:
                                    self.TruePositive[k] += 1
                                else:
                                    self.FalseNegative[k] += 1
                            else: 
                                if PredLabel[i].item() == k:
                                    self.FalsePositive[k] += 1
                                else:
                                    self.TrueNegative[k] += 1
    
        elif 'segmentation' in self.opt.task:
            AreaInter, AreaUnion = computeIoUBatch(PredLabel, TargetLabel)
            
            UnionIdx = np.where(AreaUnion > 1e-5)
            self.UnionCout[UnionIdx] += 1
            
            self.AreaInter += AreaInter
            self.AreaUnion += AreaUnion
        
    @torch.inference_mode()
    def update(self):
        self.Loss = self.RunningLoss / max(self.NumTotal, 1) # for val start point
        self.LossList.append(self.Loss)
        
        if 'classification' in self.opt.task:
            self.Top1Accuracy = self.NumTop1Correct / self.NumTotal * self.Scale  
            self.Top1AccuracyList.append(self.Top1Accuracy)
            if not self.opt.sup_metrics:
                self.Top5Accuracy = self.NumTop5Correct / self.NumTotal * self.Scale
                self.Top5AccuracyList.append(self.Top5Accuracy)
            
            self.ComMetric = self.Top1Accuracy
            
            if 'val' in self.State and self.opt.sup_metrics:
                self.Recall, self.Precision, self.Specificity, self.F1Score = \
                    computeClsValMetric(self.TruePositive, self.TrueNegative, self.FalsePositive, self.FalseNegative)
                    
                self.Recall *= self.Scale
                self.Precision *= self.Scale
                self.Specificity *= self.Scale
                self.F1Score *= self.Scale
                
                self.RecallDF = pd.concat([self.RecallDF, pd.DataFrame(self.Recall).T])
                self.PrecisionDF = pd.concat([self.PrecisionDF, pd.DataFrame(self.Precision).T])
                self.F1ScoreDF = pd.concat([self.F1ScoreDF, pd.DataFrame(self.F1Score).T])
                self.SpecificityDF = pd.concat([self.SpecificityDF, pd.DataFrame(self.Specificity).T])
                
                if "macro" in self.opt.metric_type :
                    # 'macro'
                    self.AvgRecall = np.mean(self.Recall)
                    self.AvgPrecision = np.mean(self.Precision)
                    self.AvgF1Score = np.mean(self.F1Score)
                    self.AvgSpecificity = np.mean(self.Specificity)
                else:
                    # "micro"
                    # AmtData = MyDataset.__len__()
                    # NumEachTestClass = MyDataset.getNumEachClass().T
                    AmtData = np.sum(self.NumEachClass)
                    self.AvgRecall = np.sum(self.Recall * self.NumEachClass.T) / AmtData
                    self.AvgPrecision = np.sum(self.Precision * self.NumEachClass.T) / AmtData
                    self.AvgF1Score = np.sum(self.F1Score * self.NumEachClass.T) / AmtData
                    self.AvgSpecificity = np.sum(self.Specificity * self.NumEachClass.T) / AmtData
                    
                self.AvgRecallList.append(self.AvgRecall)
                self.AvgPrecisionList.append(self.AvgPrecision)
                self.AvgF1ScoreList.append(self.AvgF1Score)
                self.AvgSpecificityList.append(self.AvgSpecificity)
                        
        elif 'segmentation' in self.opt.task:
            self.IoU = self.AreaInter / self.AreaUnion
            # self.IoU += np.expand_dims(self.AreaInter / self.AreaUnion, axis=1)
            
            self.IoUDF = pd.concat([self.IoUDF, pd.DataFrame(self.IoU * self.Scale).T])
            
            if "macro" in self.opt.metric_type:
                self.mIoU = np.mean(self.IoU) * self.Scale
            else:
                self.mIoU = np.sum(self.IoU * self.UnionCout.T) / np.sum(self.UnionCout) * self.Scale
                
            self.ComMetric = self.mIoU
            
            self.mIoUList.append(self.mIoU)
