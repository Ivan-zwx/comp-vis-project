import torch
from src.pipeline.evaluation.model_metrics import dice_coefficient, iou_score

from src.config.parameters import LOSS_CONFIG


# Model Evaluation processing the predictions already made
def evaluate_model(all_true_masks, all_pred_masks, criterion, device):
    total_loss = 0
    # Move predictions to device for loss calculation
    all_pred_masks = all_pred_masks.to(device)
    all_true_masks = all_true_masks.to(device)

    for true_masks, pred_masks in zip(all_true_masks, all_pred_masks):
        loss = criterion(pred_masks, true_masks)
        total_loss += loss.item()

    return total_loss / len(all_true_masks)


def evaluate_metrics(true_masks, pred_masks):
    """
    Computes the average Dice coefficient and IoU score across the dataset.
    Applies a threshold of 0.5 on the sigmoid of pred_masks.
    """
    # Convert raw outputs to binary predictions
    threshold = LOSS_CONFIG["threshold"]  # 0.5
    # Convert raw outputs to binary predictions using the configured threshold
    pred_masks_bin = (torch.sigmoid(pred_masks) > threshold).float()
    dice_scores = []
    iou_scores = []
    # Loop over each mask pair
    for true, pred in zip(true_masks, pred_masks_bin):
        dice_scores.append(dice_coefficient(pred, true).item())
        iou_scores.append(iou_score(pred, true).item())
    avg_dice = sum(dice_scores) / len(dice_scores)
    avg_iou = sum(iou_scores) / len(iou_scores)
    return avg_dice, avg_iou
