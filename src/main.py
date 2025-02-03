import os

import torch
import torch.nn as nn

from src.pipeline.evaluation.model_evaluation import evaluate_model
from src.pipeline.evaluation.model_visualization import visualize_results
from src.pipeline.inference.model_inference import infer
from src.pipeline.training.model_optimizer import get_optimizer
from src.pipeline.training.model_training import train_model
from src.utils.project_directories import get_data_dir_str, get_model_dir_str
from src.pipeline.data.data_loader import get_data_loader
from src.pipeline.model.segmentation_model import get_model


# PIPELINE:
# segmentation_dataset & mask_normalization >>> data_loader & data_transform
# segmentation_model & model_inference >>> model_evaluation & model_visualization

# FINE-TUNING:
# model_optimizer & model_training


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

root_dir = get_data_dir_str()
model_dir = get_model_dir_str()


def model_fine_tuning(criterion, model_path=''):
    # Setup DataLoader
    data_loader = get_data_loader(root_dir)

    # Load and prepare Model
    model = get_model(device)

    # Get Optimizer and Loss Function
    optimizer = get_optimizer(model)
    # criterion = nn.BCEWithLogitsLoss()    # binary classification metric
    # criterion = nn.MSELoss(reduction="mean")  # ALTERNATIVE - MATEO

    # Train the Model
    train_model(model, data_loader, criterion, optimizer, device)

    # Save Fine-Tuned Model Parameters
    model_filename = 'unet_vegetable_segmentation_mse.pth'
    model_path = os.path.join(model_dir, model_filename)

    torch.save(model.state_dict(), model_path)


def model_inference(criterion, load_model_params=False, model_path=''):
    # Setup DataLoader
    data_loader = get_data_loader(root_dir)

    # Load Model
    model = get_model(device)

    if load_model_params:
        # Load Fine-Tuned Model Parameters
        model.load_state_dict(torch.load(model_path))
        model.to(device)

    # Perform Full Dataset Inference
    images, true_masks, pred_masks = infer(model, data_loader, device)

    # Define Loss Function and Evaluate Model using accumulated data
    # criterion = nn.BCEWithLogitsLoss()  # binary classification metric
    # criterion = nn.MSELoss(reduction="mean")  # ALTERNATIVE - MATEO
    # criterion = nn.CrossEntropyLoss()   # multiclass classification metric
    average_loss = evaluate_model(true_masks, pred_masks, criterion, device)
    print(f'Average loss: {average_loss}')

    # Visualization of results
    visualize_results(images, true_masks, pred_masks)


# Main Function
if __name__ == '__main__':
    model_filename = 'unet_vegetable_segmentation.pth'
    model_path = os.path.join(model_dir, model_filename)

    criterion = nn.BCEWithLogitsLoss()

    # model_fine_tuning(criterion=criterion, model_path=model_path)
    model_inference(criterion=criterion, load_model_params=False, model_path=model_path)

    pass
