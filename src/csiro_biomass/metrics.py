import torch


def compute_weighted_r2(pred, target):
    weights = torch.Tensor([1, 1, 1]).to(pred.device)

    target = target.to(pred.device)

    col_means = target.mean(dim=0)
    ss_res = ((target - pred) ** 2).sum(dim=0)

    # Avoid division by zero
    ss_tot = ((target - col_means) ** 2).sum(dim=0)
    ss_tot = torch.clamp(ss_tot, min=1e-8)

    r2_per_col = 1 - (ss_res / ss_tot)
    score = (weights * r2_per_col).mean()

    return score
