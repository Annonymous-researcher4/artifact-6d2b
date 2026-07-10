import torch


def calc_precision(preds, labels):
    if (preds == 1).sum().item() == 0:
        return torch.tensor([0.0])
    prec = ((preds == 1) & (preds == labels)).sum() / (preds == 1).sum()
    return prec


def calc_recall(preds, labels):
    if (labels == 1).sum().item() == 0:
        return torch.tensor([0.0])
    rec = ((labels == 1) & (preds == labels)).sum() / (labels == 1).sum()
    return rec


def calc_F1(preds, labels):
    if (preds == 1).sum().item() == 0:
        return torch.tensor([0.0])
    rec = calc_recall(preds, labels)
    prec = calc_precision(preds, labels)
    if (prec + rec) == 0:
        return torch.tensor([0.0])
    return (2 * prec * rec) / (prec + rec)
