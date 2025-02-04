from PIL import Image
import numpy as np

from src.utils.project_directories import get_data_dir_str
from src.utils.data_subdirectories import get_relative_mask_dir


def analyze_segmentation_mask_values(mask_image_path):
    # Load the mask image
    mask = Image.open(mask_image_path).convert('L')  # Ensure it's in grayscale

    # Convert to a numpy array and analyze unique values
    mask_array = np.array(mask)
    unique_values = np.unique(mask_array)
    print("Unique values in the mask:", unique_values)


if __name__ == '__main__':
    root_dir = get_data_dir_str()
    train_masks_dir = get_relative_mask_dir(root_dir)

    segmentation_masks = [
        '\\0cdf5b5d0ce1_01_mask.gif',
        '\\0ce66b539f52_14_mask.gif',
        '\\1a17a1bd648b_13_mask.gif',
        '\\2af7c265531e_04_mask.gif',
        '\\3f8d611822bc_13_mask.gif'
    ]

    for segmentation_mask in segmentation_masks:
        analyze_segmentation_mask_values(train_masks_dir + segmentation_mask)
