import segmentation_models_pytorch as smp
import torch

from src.pipeline.model.custom_model import CustomUNet

from src.config.parameters import MODEL_CONFIG


# Load Pretrained Model
def get_model(device, use_custom_model=False):
    # Create a U-Net model pre-trained on ImageNet
    if not use_custom_model:
        model = smp.Unet(
            encoder_name=MODEL_CONFIG["encoder_name"],  # "resnet34"  # Using ResNet-34 as the encoder.
            encoder_weights=MODEL_CONFIG["encoder_weights"],  # "imagenet"  # Initialize with pre-trained weights on ImageNet.
            in_channels=MODEL_CONFIG["in_channels"],  # 3  # Number of input channels (e.g., RGB).
            classes=MODEL_CONFIG["classes"]  # 1  # One output channel for binary classification.
        )
    else:
        model = CustomUNet(
            in_channels=MODEL_CONFIG["in_channels"],  # 3  # Number of input channels (e.g., RGB).
            out_channels=MODEL_CONFIG["classes"]  # 1  # One output channel for binary classification.
        )
    return model.to(device)
