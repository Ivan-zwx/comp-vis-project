import os
from pathlib import Path

from src.utils.project_directories import get_results_dir


def get_resnet34_results_subdirectory():
    results_dir = get_results_dir()
    resnet34_subdir = results_dir / "resnet34"
    return resnet34_subdir


def get_efficientnet_b0_results_results_subdirectory():
    results_dir = get_results_dir()
    efficientnet_b0_subdir = results_dir / "efficientnet-b0"
    return efficientnet_b0_subdir


# def get_model3_results_results_subdirectory():
#     results_dir = get_results_dir()
#     resnet34_subdir = results_dir / "model3"
#     return resnet34_subdir
