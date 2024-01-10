import torch


def topKCorrect(Prediction, Target, Topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(Topk)

        _, Pred = Prediction.topk(maxk, 1, True, True)
        Pred = Pred.t()
        Correct = Pred.eq(Target.view(1, -1).expand_as(Pred))

        CorrectKList = []
        for k in Topk:
            CorrectK = Correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            CorrectKList.append(CorrectK.cpu().numpy()[0])
        return CorrectKList