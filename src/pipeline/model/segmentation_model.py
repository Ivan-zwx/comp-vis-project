import segmentation_models_pytorch as smp
import torch


# Load Pretrained Model
def get_model(device):
    # Create a U-Net model pre-trained on ImageNet
    model = smp.Unet(
        encoder_name="resnet34",      # Using ResNet-34 as the encoder.
        encoder_weights="imagenet",   # Initialize with pre-trained weights on ImageNet.
        in_channels=3,                # Number of input channels (e.g., RGB).
        classes=1                     # One output channel for binary classification.
    )
    return model.to(device)
