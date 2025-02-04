import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
from src.pipeline.evaluation.model_metrics import dice_coefficient, iou_score

from src.utils.model_subdirectories import get_model_checkpoint_path

from src.config.parameters import TRAINING_CONFIG, LOSS_CONFIG


def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                num_epochs=TRAINING_CONFIG["num_epochs"],  # 10
                patience=TRAINING_CONFIG["patience"],  # 3
                checkpoint_path=get_model_checkpoint_path()):
    """
    Train the model with early stopping and checkpointing based on validation loss.

    Args:
        model: The segmentation model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer for updating model parameters.
        device: The computation device (CPU or GPU).
        num_epochs: Maximum number of training epochs.
        patience: Number of epochs to wait for improvement before stopping early.
        checkpoint_path: File path to save the best model checkpoint.

    Returns:
        A tuple of lists: (epoch_losses, val_losses, val_dice_scores, val_iou_scores)
    """

    model.train()  # Set model to training mode
    epoch_losses = []  # List to store training loss per epoch
    val_losses = []  # List to store validation loss per epoch
    val_dice_scores = []  # List to store average validation Dice coefficient per epoch
    val_iou_scores = []  # List to store average validation IoU score per epoch

    best_val_loss = float('inf')  # Initialize best validation loss as infinity
    epochs_without_improvement = 0  # Counter for epochs without improvement

    for epoch in range(num_epochs):
        running_loss = 0.0  # Accumulate training loss over batches
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()  # Zero gradients before each batch
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, masks)  # Compute loss
            loss.backward()  # Backward pass (compute gradients)
            optimizer.step()  # Update model parameters

            # Accumulate loss weighted by batch size
            running_loss += loss.item() * images.size(0)

        # Compute average training loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')

        # -------- Validation Phase --------
        model.eval()  # Set the model to evaluation mode for validation
        running_val_loss = 0.0  # Accumulate validation loss
        dice_scores = []  # List for Dice scores on validation batches
        iou_scores = []  # List for IoU scores on validation batches
        with torch.no_grad():
            for val_images, val_masks in val_loader:
                val_images = val_images.to(device)
                val_masks = val_masks.to(device)
                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_masks)
                running_val_loss += val_loss.item() * val_images.size(0)

                # Calculate metrics per batch using the threshold from LOSS_CONFIG
                threshold = LOSS_CONFIG["threshold"]  # e.g., 0.5
                pred_bin = (torch.sigmoid(val_outputs) > threshold).float()
                for true, pred in zip(val_masks, pred_bin):
                    dice_scores.append(dice_coefficient(pred, true).item())
                    iou_scores.append(iou_score(pred, true).item())

        # Compute average validation loss and metrics
        avg_val_loss = running_val_loss / len(val_loader.dataset)
        avg_dice = sum(dice_scores) / len(dice_scores)
        avg_iou = sum(iou_scores) / len(iou_scores)
        val_losses.append(avg_val_loss)
        val_dice_scores.append(avg_dice)
        val_iou_scores.append(avg_iou)
        print(f'Validation Loss: {avg_val_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}')

        # -------- Early Stopping and Checkpointing --------
        # Check if the validation loss improved compared to the best seen so far.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss  # Update the best validation loss
            epochs_without_improvement = 0  # Reset the counter
            # Save the current model parameters as a checkpoint
            torch.save(model.state_dict(), checkpoint_path)
            print("Validation loss improved. Saving new best model checkpoint.")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

        # If no improvement has been seen for 'patience' consecutive epochs, stop training early.
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

        model.train()  # Switch back to training mode for the next epoch

    print('Finished Training')
    return epoch_losses, val_losses, val_dice_scores, val_iou_scores
