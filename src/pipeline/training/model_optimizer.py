import torch.optim as optim

from src.config.parameters import TRAINING_CONFIG


def get_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])  # 0.001  # Learning rate might need tuning
    return optimizer
