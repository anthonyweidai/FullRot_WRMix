import numpy as np

from ..data import CustomDataset

def metricMonitor(Recall, Precision, F1Score, Specificity,
                  MyDataset: CustomDataset=None, MetricsType="micro"):
    if "macro" in MetricsType :
        # 'macro'
        AvgRecall = np.mean(Recall)
        AvgPrecision = np.mean(Precision)
        AvgF1Score = np.mean(F1Score)
        AvgSpecificity = np.mean(Specificity)
    else:
        # "micro"
        AmtData = MyDataset.__len__()
        NumEachTestClass = MyDataset.getNumEachClass().T
        AvgRecall = np.sum(Recall * NumEachTestClass) / AmtData
        AvgPrecision = np.sum(Precision * NumEachTestClass) / AmtData
        AvgF1Score = np.sum(F1Score * NumEachTestClass) / AmtData
        AvgSpecificity = np.sum(Specificity * NumEachTestClass) / AmtData

    return AvgRecall, AvgPrecision, AvgF1Score, AvgSpecificity
