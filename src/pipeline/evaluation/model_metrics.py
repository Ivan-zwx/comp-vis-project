import torch


def dice_coefficient(pred, target, smooth=1.0):
    """
    Computes the Dice coefficient between two binary masks.
    Assumes pred and target are tensors of the same shape.
    """
    pred = pred.float().view(-1)
    target = target.float().view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, smooth=1.0):
    """
    Computes the Intersection over Union (IoU) score between two binary masks.
    """
    pred = pred.float().view(-1)
    target = target.float().view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)
