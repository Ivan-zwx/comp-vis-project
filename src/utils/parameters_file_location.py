import os
from pathlib import Path

from src.utils.project_directories import get_project_dir


def get_parameters_file_location():
    project_dir = get_project_dir()
    parameters_file_location = project_dir / "src" / "config" / "parameters.py"
    return parameters_file_location
