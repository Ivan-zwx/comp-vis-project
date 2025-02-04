from torch.utils.data import DataLoader, random_split
import torch

from src.pipeline.data.data_transform import get_data_transform
from src.pipeline.data.segmentation_dataset import SegmentationDataset


# Data Loader and Transformations
def get_data_loader(root_dir, transform=get_data_transform(), batch_size=50, shuffle=True,
                    # GPU optimizations
                    pin_memory=True, num_workers=10):

    dataset = SegmentationDataset(root_dir, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=pin_memory, num_workers=num_workers)


def get_train_val_loaders(root_dir, transform=get_data_transform(), train_split=0.8, batch_size=50,
                          # Fixed Random Seed (Train/Validation Split)
                          seed=42,
                          # GPU Optimizations
                          pin_memory=True, num_workers=10):

    dataset = SegmentationDataset(root_dir, transform=transform)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    # Create a generator with a fixed seed
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=pin_memory, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=pin_memory, num_workers=num_workers)

    return train_loader, val_loader
