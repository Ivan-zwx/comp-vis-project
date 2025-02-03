from torch.utils.data import DataLoader, random_split

from src.pipeline.data.data_transform import get_data_transform
from src.pipeline.data.segmentation_dataset import SegmentationDataset


# Data Loader and Transformations
def get_data_loader(root_dir, transform=get_data_transform(), batch_size=50, shuffle=True,
                    # GPU optimizations
                    pin_memory=True, num_workers=10):
    dataset = SegmentationDataset(root_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=pin_memory, num_workers=num_workers)


def get_train_val_loaders(root_dir, transform=get_data_transform(), train_split=0.8, batch_size=5, num_workers=5):
    dataset = SegmentationDataset(root_dir, transform=transform)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
