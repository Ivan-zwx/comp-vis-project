import csv


def log_training_results(epoch_losses, val_losses, val_dice_scores, val_iou_scores, log_path):
    """
    Logs training and validation results to a CSV file.

    Parameters:
    - epoch_losses (list): Training loss per epoch.
    - val_losses (list): Validation loss per epoch.
    - val_dice_scores (list): Validation Dice coefficient per epoch.
    - val_iou_scores (list): Validation IoU score per epoch.
    - log_path (str): File path to save the CSV file.
    """

    with open(log_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val Dice", "Val IoU"])

        # Write the data for each epoch
        for i in range(len(epoch_losses)):
            writer.writerow([
                i + 1,  # Epoch number
                f"{epoch_losses[i]:.6f}",
                f"{val_losses[i]:.6f}",
                f"{val_dice_scores[i]:.6f}",
                f"{val_iou_scores[i]:.6f}"
            ])
