import os

import torch
import torch.nn as nn

from src.pipeline.evaluation.model_evaluation import evaluate_model, evaluate_metrics
from src.pipeline.evaluation.model_visualization import visualize_inference_results, visualize_training_results
from src.pipeline.inference.model_inference import infer
from src.pipeline.training.model_optimizer import get_optimizer
from src.pipeline.training.model_training import train_model
from src.utils.project_directories import get_data_dir_str, get_model_dir_str
from src.pipeline.data.data_loader import get_data_loader, get_train_val_loaders
from src.pipeline.model.segmentation_model import get_model
from src.utils.model_subdirectories import get_model_checkpoint_path

from src.config.parameters import TRAINING_CONFIG


# PIPELINE:
# segmentation_dataset & mask_normalization >>> data_loader & data_transform
# segmentation_model & model_inference >>> model_evaluation & model_visualization

# FINE-TUNING:
# model_optimizer & model_training


# Determine device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


root_dir = get_data_dir_str()
model_dir = get_model_dir_str()


def model_fine_tuning(criterion, model_path=None):
    # Setup DataLoader: split the dataset into train and validation
    # data_loader = get_data_loader(root_dir, batch_size=50)
    train_loader, val_loader = get_train_val_loaders(root_dir)  # , batch_size=50, seed=42

    # Load and prepare Model
    model = get_model(device)

    # Get Optimizer and Loss Function
    optimizer = get_optimizer(model)
    # criterion = nn.BCEWithLogitsLoss()    # binary classification metric
    # criterion = nn.MSELoss(reduction="mean")  # ALTERNATIVE - MATEO

    # Train the Model
    # train_model(model, data_loader, criterion, optimizer, device)

    # Train the Model and capture loss history
    # epoch_losses = train_model(model, data_loader, criterion, optimizer, device, num_epochs=10)

    # Number of epochs for training (and visualization of training performance)
    # num_epochs = 10  # Obsolete - now in parameters file

    # Train the model with validation tracking
    epoch_losses, val_losses, val_dice_scores, val_iou_scores = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,  # num_epochs=num_epochs  # 10 epochs by default
    )

    # Save Fine-Tuned Model Parameters
    # model_filename = 'carvana_model_4.pth'  # Redundant and obsolete - passed as function parameter
    # model_path = os.path.join(model_dir, model_filename)  # Redundant and obsolete - passed as function parameter

    # Check if a checkpoint exists (which indicates training was interrupted early or ended with early stopping).
    checkpoint_path = get_model_checkpoint_path()
    if os.path.exists(checkpoint_path):
        # Load the best model checkpoint into the current model.
        model.load_state_dict(torch.load(checkpoint_path))
        print("Loaded best checkpoint from training.")
    else:
        print("No checkpoint found; using the current model state.")

    if model_path:
        torch.save(model.state_dict(), model_path)

    # Delete the checkpoint file to avoid interference with future training runs.
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint file deleted.")

    # Visualization of model parameter changes per epoch
    visualize_training_results(epoch_losses, val_losses, val_dice_scores, val_iou_scores)  # , num_epochs=num_epochs

    # return model, epoch_losses, val_losses, val_dice_scores, val_iou_scores  # No need to return anything here


def model_inference(criterion, load_model_params=False, model_path=''):
    # Setup DataLoader
    data_loader = get_data_loader(root_dir)  # , batch_size=50

    # Load Model
    model = get_model(device)

    if load_model_params:
        # Load Fine-Tuned Model Parameters
        model.load_state_dict(torch.load(model_path))

    # Ensure model is on GPU
    # model.to(device)  # Redundant - already done in get_model(device)

    # Perform Full Dataset Inference
    images, true_masks, pred_masks = infer(model, data_loader, device)

    # Define Loss Function and Evaluate Model using accumulated data
    # criterion = nn.BCEWithLogitsLoss()  # binary classification metric
    # criterion = nn.MSELoss(reduction="mean")  # ALTERNATIVE - MATEO
    # criterion = nn.CrossEntropyLoss()   # multiclass classification metric
    average_loss = evaluate_model(true_masks, pred_masks, criterion, device)
    # print(f'Average loss: {average_loss}')

    # Compute Dice and IoU scores
    avg_dice, avg_iou = evaluate_metrics(true_masks, pred_masks)

    print(f'Average Loss: {average_loss:.4f}')
    print(f'Average Dice Score: {avg_dice:.4f}')
    print(f'Average IoU Score: {avg_iou:.4f}')

    # Visualization of results
    visualize_inference_results(images, true_masks, pred_masks)

    # return images, true_masks, pred_masks  # No need to return anything here


# Main Function
if __name__ == '__main__':
    model_filename = 'carvana_model_5.pth'
    model_path = os.path.join(model_dir, model_filename)

    criterion = nn.BCEWithLogitsLoss()

    model_fine_tuning(criterion=criterion, model_path=model_path)
    # model_inference(criterion=criterion, load_model_params=True, model_path=model_path)

    pass
