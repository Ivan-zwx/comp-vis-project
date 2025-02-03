from torchvision import transforms


def get_data_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images and masks to 256x256 pixels
        transforms.ToTensor()           # Convert images and masks to torch tensors
    ])
