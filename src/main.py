import os
import torch
import torch.nn as nn

# Import evaluation and visualization functions for both inference and training.
from src.pipeline.evaluation.model_evaluation import evaluate_model, evaluate_metrics
from src.pipeline.evaluation.model_visualization import visualize_inference_results, visualize_training_results
from src.pipeline.evaluation.model_logging import log_training_results
from src.config.parameter_logging import log_parameters
# Import the inference, optimizer, training, and data loader functions.
from src.pipeline.inference.model_inference import infer
from src.pipeline.training.model_optimizer import get_optimizer
from src.pipeline.training.model_training import train_model
from src.pipeline.data.data_loader import get_data_loader, get_train_val_loaders
# Import the model constructor.
from src.pipeline.model.segmentation_model import get_model
# Import project directory utilities.
from src.utils.project_directories import get_data_dir_str, get_model_dir_str
from src.utils.model_subdirectories import get_model_checkpoint_path
from src.utils.results_subdirectories import (get_resnet34_results_subdirectory,
                                              get_efficientnet_b0_results_results_subdirectory)

# Determine the computation device: use GPU if available, else CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Set the root directory for data and the model directory.
root_dir = get_data_dir_str()  # Assumes data directory (e.g., data/carvana)
model_dir = get_model_dir_str()  # Directory for saving models


def model_fine_tuning(criterion, model_path=None):
    """
    Fine-tunes the model using a training/validation split with early stopping and checkpointing.
    Loads the best model checkpoint (if training was interrupted early or early stopping occurred)
    and saves it to the specified model_path. Finally, it visualizes the training progress.

    Args:
        criterion: Loss function to be used.
        model_path: Full path (including filename) where the best model will be saved.
    """
    # Split the dataset into training and validation subsets.
    train_loader, val_loader = get_train_val_loaders(root_dir)

    # Initialize the model and move it to the selected device.
    model = get_model(device)

    # Get the optimizer (configured using TRAINING_CONFIG parameters).
    optimizer = get_optimizer(model)

    # Train the model while capturing training and validation metrics.
    # The train_model function handles early stopping and checkpointing.
    epoch_losses, val_losses, val_dice_scores, val_iou_scores = train_model(
        model, train_loader, val_loader, criterion, optimizer, device
    )

    # Define the checkpoint file path used during training.
    checkpoint_path = get_model_checkpoint_path()

    # Load the best model checkpoint (if it exists) so that we use the best performing model.
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print("Loaded best checkpoint from training.")
    else:
        print("No checkpoint found; using the current model state.")

    # Save the best model to the specified model_path.
    if model_path:
        torch.save(model.state_dict(), model_path)
        print(f"Best model saved to {model_path}")

    # Delete the checkpoint file to avoid interference with future training runs.
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint file deleted.")

    # Visualize training progress: loss curves and validation metrics.
    visualize_training_results(epoch_losses, val_losses, val_dice_scores, val_iou_scores)

    # SAVE THE TRAINING PERFORMANCE OF RESNET_34 BACKBONE U-NET
    # results_dir = get_resnet34_results_subdirectory()
    # plot_file = os.path.join(results_dir, "resnet34_3_training_plot.png")
    # log_file = os.path.join(results_dir, "resnet34_3_training_results.csv")
    # params_file = os.path.join(results_dir, "resnet34_3_training_params.txt")
    # visualize_training_results(epoch_losses, val_losses, val_dice_scores, val_iou_scores,
    #                            plot_path=plot_file)
    # log_training_results(epoch_losses, val_losses, val_dice_scores, val_iou_scores,
    #                      log_path=log_file)
    # log_parameters(params_file)

    # SAVE THE TRAINING PERFORMANCE OF EFFICIENTNET-B0 BACKBONE U-NET
    # results_dir = get_efficientnet_b0_results_results_subdirectory()
    # plot_file = os.path.join(results_dir, "efficientnet-b0_1_training_plot.png")
    # log_file = os.path.join(results_dir, "efficientnet-b0_1_training_results.csv")
    # params_file = os.path.join(results_dir, "efficientnet-b0_1_training_params.txt")
    # visualize_training_results(epoch_losses, val_losses, val_dice_scores, val_iou_scores,
    #                            plot_path=plot_file)
    # log_training_results(epoch_losses, val_losses, val_dice_scores, val_iou_scores,
    #                      log_path=log_file)
    # log_parameters(params_file)


def model_inference(criterion, model_path=None):
    """
    Performs inference using the trained model and computes evaluation metrics.
    It prints the average loss, Dice, and IoU scores and visualizes the inference results.

    Args:
        criterion: Loss function to use for evaluation.
        model_path: Path to the saved model parameters.
    """
    # Setup DataLoader for inference (no shuffling needed).
    # data_loader = get_data_loader(root_dir, shuffle=False)
    train_loader, val_loader = get_train_val_loaders(root_dir)
    data_loader = val_loader

    # Initialize the model.
    model = get_model(device)

    # Load saved model parameters if requested.
    if model_path:
        model.load_state_dict(torch.load(model_path))

    # Run inference on the entire dataset.
    images, true_masks, pred_masks = infer(model, data_loader, device)

    # Evaluate model performance on the inference results.
    average_loss = evaluate_model(true_masks, pred_masks, criterion, device)
    avg_dice, avg_iou = evaluate_metrics(true_masks, pred_masks)
    print(f'Average Loss: {average_loss:.4f}')
    print(f'Average Dice Score: {avg_dice:.4f}')
    print(f'Average IoU Score: {avg_iou:.4f}')

    # Visualize the inference results.
    visualize_inference_results(images, true_masks, pred_masks)


# Main function to execute training or inference.
if __name__ == '__main__':
    # Define the filename and full path for saving the fine-tuned model.
    model_filename = 'carvana_efficientnet-b0_1.pth'
    model_path = os.path.join(model_dir, model_filename)

    # Define the loss function for binary segmentation.
    criterion = nn.BCEWithLogitsLoss()

    # Run the fine-tuning (training) process.
    model_fine_tuning(criterion=criterion, model_path=model_path)

    # Optionally, run inference using the saved model.
    # model_inference(criterion=criterion, model_path=model_path)

    pass
