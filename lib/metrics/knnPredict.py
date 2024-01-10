import torch


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knnPredict(opt, Feature, FeatureBank, FeatureLabels):
    K = opt.knn_k
    T = 0.1
    Classes = opt.cls_num_classes
    
    # compute cos similarity between each Feature vector and Feature bank ---> [B, N]
    sim_matrix = torch.mm(Feature, FeatureBank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=K, dim=-1)
    # [B, K]
    sim_labels = torch.gather(FeatureLabels.expand(Feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / T).exp()

    # counts for each class
    one_hot_label = torch.zeros(Feature.size(0) * K, Classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(Feature.size(0), -1, Classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels