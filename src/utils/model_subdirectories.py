import os
from pathlib import Path

from src.utils.project_directories import get_model_dir_str


def get_model_checkpoint_dir():
    model_dir = get_model_dir_str()
    model_checkpoint_dir = os.path.join(model_dir, "checkpoint")
    return model_checkpoint_dir


def get_model_checkpoint_path():
    model_checkpoint_dir = get_model_checkpoint_dir()
    checkpoint_path = os.path.join(model_checkpoint_dir, "best_model_checkpoint.pth")
    return checkpoint_path
