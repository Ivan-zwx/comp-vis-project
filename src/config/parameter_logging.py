import importlib.util

from src.utils.parameters_file_location import get_parameters_file_location


def load_config_from_file(file_path):
    """
    Dynamically loads all dictionary variables from a Python file, excluding built-ins and special attributes.

    Parameters:
    - file_path (str): Path to the Python file containing configurations.

    Returns:
    - dict: Dictionary containing only user-defined dictionary-type configurations.
    """
    spec = importlib.util.spec_from_file_location("parameters", file_path)
    parameters = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parameters)

    # Extract only user-defined dictionaries (exclude built-ins and special attributes)
    return {
        key: value for key, value in vars(parameters).items()
        if isinstance(value, dict) and not key.startswith("__")
    }


def log_parameters(log_path, file_path=get_parameters_file_location()):
    """
    Loads configuration dictionaries from a Python file and logs them to a text file.

    Parameters:
    - file_path (str): Path to the Python file containing configuration dictionaries.
    - log_path (str): File path to save the parameters log.
    """
    config_dict = load_config_from_file(file_path)

    with open(log_path, "w") as file:
        for section, params in config_dict.items():
            file.write(f"[{section}]\n")
            for key, value in params.items():
                file.write(f"{key} = {value}\n")
            file.write("\n")  # Add spacing between sections


# **One-liner function call to log all dictionaries from parameters.py**
log_parameters("parameters_log.txt")
