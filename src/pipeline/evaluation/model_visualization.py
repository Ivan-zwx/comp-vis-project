import torch
import matplotlib.pyplot as plt
import numpy as np


# Visualization Function
def visualize_results(images, true_masks, pred_masks):
    images = images.cpu().numpy()
    true_masks = true_masks.cpu().numpy()
    pred_masks = torch.sigmoid(pred_masks).cpu().numpy() > 0.5  # Applying threshold to get binary masks

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
