import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from src.config.parameters import LOSS_CONFIG, TRAINING_CONFIG


# Visualization Function
def visualize_inference_results(images, true_masks, pred_masks):
    images = images.cpu().numpy()
    true_masks = true_masks.cpu().numpy()
    threshold = LOSS_CONFIG["threshold"]  # 0.5
    pred_masks = torch.sigmoid(pred_masks).cpu().numpy() > threshold  # Applying threshold to get binary masks

    n_images = 10  # Assuming this is your batch size, adjust if different

    # Create 3 times n_images subplots (3 per image: original, true mask, predicted mask)
    fig, axes = plt.subplots(nrows=n_images, ncols=3, figsize=(15, 2 * n_images))  # Adjust figsize if necessary

    for i in range(n_images):
        # Original Image
        axes[i, 0].imshow(np.transpose(images[i], (1, 2, 0)))  # CHW to HWC
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        # True Mask
        axes[i, 1].imshow(true_masks[i].squeeze(), cmap='gray')  # Squeeze to remove channel dim if it's 1
        axes[i, 1].set_title('True Mask')
        axes[i, 1].axis('off')

        # Predicted Mask
        visible_pred_mask = pred_masks[i].squeeze() * 255  # Scale for visibility, squeeze to remove channel dim
        axes[i, 2].imshow(visible_pred_mask, cmap='gray')
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')

    plt.tight_layout()  # Improve spacing between plots
    plt.show()


def visualize_training_results(epoch_losses, val_losses, val_dice_scores, val_iou_scores, num_epochs=TRAINING_CONFIG["num_epochs"]):  # num_epochs=10

    # Plot training and validation curves
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 4))

    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, epoch_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses, marker='o', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    plt.grid(True)

    # Dice plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_dice_scores, marker='o', label='Val Dice', color='g')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.title('Validation Dice vs. Epochs')
    plt.legend()
    plt.grid(True)

    # IoU plot
    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_iou_scores, marker='o', label='Val IoU', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('IoU Score')
    plt.title('Validation IoU vs. Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()