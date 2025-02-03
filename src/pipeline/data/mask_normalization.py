import numpy as np
import torch


def normalize_mask(mask):
    mask_array = np.array(mask)
    normalized_mask = np.where(mask_array > 0, 1, 0).astype(np.float32)  # Normalize mask
    mask = torch.from_numpy(normalized_mask)    # .unsqueeze(0)   # Add channel dimension
    # Remove any unnecessary singleton dimensions
    # mask = torch.squeeze(mask, 0)   # Squeeze out the singleton dimension if it's size 1
    return mask
