import torch


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
