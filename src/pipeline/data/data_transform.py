from torchvision import transforms

from src.config.parameters import TRANSFORM_CONFIG


def get_data_transform():
    return transforms.Compose([
        transforms.Resize(TRANSFORM_CONFIG["resize"]),  # (256, 256)  # Resize images and masks to 256x256 pixels
        transforms.ToTensor()           # Convert images and masks to torch tensors
    ])


def get_manual_augment_transform():
    return transforms.Compose([
        transforms.Resize(TRANSFORM_CONFIG["resize"]),  # Ensure consistent size
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomRotation(15),  # Random rotation by up to 15 degrees
        transforms.ToTensor()
    ])