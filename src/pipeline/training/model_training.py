import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from src.pipeline.evaluation.model_metrics import dice_coefficient, iou_score


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    model.train()  # Set the model to training mode

    # Track various metrics across epochs
    epoch_losses = []
    val_losses = []
    val_dice_scores = []
    val_iou_scores = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')


        # Validation step
        model.eval()  # Set the model to validation mode
        running_val_loss = 0.0
        dice_scores = []
        iou_scores = []
        with torch.no_grad():
            for val_images, val_masks in val_loader:
                val_images = val_images.to(device)
                val_masks = val_masks.to(device)
                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_masks)
                running_val_loss += val_loss.item() * val_images.size(0)

                # Calculate metrics per batch
                pred_bin = (torch.sigmoid(val_outputs) > 0.5).float()
                for true, pred in zip(val_masks, pred_bin):
                    dice_scores.append(dice_coefficient(pred, true).item())
                    iou_scores.append(iou_score(pred, true).item())

        avg_val_loss = running_val_loss / len(val_loader.dataset)
        avg_dice = sum(dice_scores) / len(dice_scores)
        avg_iou = sum(iou_scores) / len(iou_scores)
        val_losses.append(avg_val_loss)
        val_dice_scores.append(avg_dice)
        val_iou_scores.append(avg_iou)
        print(f'Validation Loss: {avg_val_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}')

        model.train()  # Set the model to training mode

    print('Finished Training')

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

    return epoch_losses, val_losses, val_dice_scores, val_iou_scores
