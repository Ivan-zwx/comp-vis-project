import os
from torch.utils.data import Dataset
from PIL import Image

from src.pipeline.data.mask_normalization import normalize_mask


# Dataset Class
class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = [file for file in os.listdir(root_dir) if '_mask' not in file]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.files[idx])
        mask_path = img_path.replace('.jpg', '_mask.png')  # Assuming images are .jpg and masks are .png

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Load mask in grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # NORMALIZING n CLASSES TO 2 CLASSES (BINARY CLASSIFICATION)
        mask = normalize_mask(mask)

        return image, mask
