from PIL import Image
import numpy as np

from src.utils.project_directories import get_data_subdir_str


def analyze_segmentation_mask_values(mask_image_path):
    # Load the mask image
    mask = Image.open(mask_image_path).convert('L')  # Ensure it's in grayscale

    # Convert to a numpy array and analyze unique values
    mask_array = np.array(mask)
    unique_values = np.unique(mask_array)
    print("Unique values in the mask:", unique_values)


if __name__ == '__main__':
    root_dir = get_data_subdir_str()

    segmentation_masks = [
        '\\bp1_jpg.rf.e42ebfba3bdf35d33357573a20219d48_mask.png',
        '\\c1_jpg.rf.2ae442e1d4b00ed5ac029ec388106172_mask.png',
        '\\cf2_jpg.rf.e1b519d6a598ea7071259b439b97598b_mask.png',
        '\\c2_jpg.rf.0dcee96e036fc79e6b148da117b29fbc_mask.png',
        '\\ep2_jpg.rf.89af8a67de58675c1073c72f017563ac_mask.png'
    ]

    for segmentation_mask in segmentation_masks:
        analyze_segmentation_mask_values(root_dir + segmentation_mask)
