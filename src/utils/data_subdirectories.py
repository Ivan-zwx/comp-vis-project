import os
from pathlib import Path


def get_relative_image_dir(root_dir):
    img_dir = os.path.join(root_dir, "train")
    img_dir = os.path.join(img_dir, "train")
    return img_dir


def get_relative_mask_dir(root_dir):
    mask_dir = os.path.join(root_dir, "train_masks")
    mask_dir = os.path.join(mask_dir, "train_masks")
    return mask_dir


def get_relative_manual_image_dir(root_dir):
    manual_img_dir = os.path.join(root_dir, "test_manual")
    return manual_img_dir


def get_relative_manual_mask_dir(root_dir):
    manual_mask_dir = os.path.join(root_dir, "test_manual_masks")
    return manual_mask_dir
