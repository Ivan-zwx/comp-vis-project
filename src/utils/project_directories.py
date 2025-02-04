from pathlib import Path


def get_project_dir():
    # Get the path of the current script
    current_file = Path(__file__).resolve()
    # To get the root directory, you may need to navigate relatively
    root_dir = current_file.parent.parent.parent
    return root_dir


def get_data_dir():
    return get_project_dir() / "data"


def get_data_dir_str():
    return str(get_data_dir())


def get_model_dir_str():
    return str(get_project_dir() / "model")


def get_results_dir():
    return get_project_dir() / "results"
