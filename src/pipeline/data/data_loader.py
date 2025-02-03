from torch.utils.data import DataLoader

from src.pipeline.data.data_transform import get_data_transform
from src.pipeline.data.segmentation_dataset import SegmentationDataset


# Data Loader and Transformations
def get_data_loader(root_dir, transform=get_data_transform(), batch_size=50, shuffle=True,
                    # GPU optimizations
                    pin_memory=True, num_workers=10):
    dataset = SegmentationDataset(root_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=pin_memory, num_workers=num_workers)
