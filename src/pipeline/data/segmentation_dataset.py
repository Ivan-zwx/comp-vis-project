import os
from torch.utils.data import Dataset
from PIL import Image

from src.pipeline.data.mask_normalization import normalize_mask


def relative_image_dir(root_dir):
    img_dir = os.path.join(root_dir, "train")
    img_dir = os.path.join(img_dir, "train")
    return img_dir


def relative_mask_dir(root_dir):
    mask_dir = os.path.join(root_dir, "train_masks")
    mask_dir = os.path.join(mask_dir, "train_masks")
    return mask_dir


# Dataset Class
class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = relative_image_dir(root_dir)
        self.mask_dir = relative_mask_dir(root_dir)
        # List only .jpg files in the image_dir
        self.files = [file for file in os.listdir(self.image_dir) if file.endswith('.jpg')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        img_path = os.path.join(self.image_dir, filename)
        # Construct mask filename: replace '.jpg' with '_mask.gif'
        mask_filename = filename.replace('.jpg', '_mask.gif')
        mask_path = os.path.join(self.mask_dir, mask_filename)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Load mask as grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Normalize the mask to 0 and 1 values (binary)
        mask = normalize_mask(mask)
        return image, mask
