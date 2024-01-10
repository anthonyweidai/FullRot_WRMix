import numpy as np


def computeClsValMetric(TruePositive, TrueNegatives, FalsePositive, FalseNegative):
    # Compute metrics: Recall, Precision, F-score, Specificity, by superposiiton
    NumClasses = TruePositive.shape[0]
    
    Recall, Precision, Specificity, F1Score = np.zeros((4, NumClasses), dtype=float)
    
    for k in range(NumClasses):
        PositiveAll = TruePositive[k] + FalseNegative[k]
        TPAndFP = TruePositive[k] + FalsePositive[k]
        NegativeAll = TrueNegatives[k] + FalsePositive[k]
        if PositiveAll != 0:
            Recall[k] = TruePositive[k] / PositiveAll
        if TPAndFP != 0:
            Precision[k] = TruePositive[k] / TPAndFP
        if (Recall[k] + Precision[k]) != 0:
            F1Score[k] = 2 * Recall[k] * Precision[k] / (Recall[k] + Precision[k])
        if NegativeAll != 0:
            Specificity[k] = TrueNegatives[k] / NegativeAll
                
    return Recall, Precision, Specificity, F1Score