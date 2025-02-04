

###############################
# Model Hyperparameters
###############################
# These parameters determine the architecture of the segmentation model.
# They are usually not tuned as hyperparameters for a given dataset—rather, they depend on your problem type.

MODEL_CONFIG = {
    # The backbone used for feature extraction.
    "encoder_name": "efficientnet-b0",  # e.g., "resnet34" is a popular choice balancing performance and speed.
    # resnet34, efficientnet-b0

    # Pre-trained weights used to initialize the encoder.
    "encoder_weights": "imagenet",  # Using weights pre-trained on ImageNet for faster convergence.

    # Number of channels in the input images.
    "in_channels": 3,  # For RGB images, this is fixed to 3; this is data-dependent.

    # Number of output channels. For binary segmentation, a single channel is used.
    "classes": 1  # For binary segmentation, one output channel suffices.
}

###############################
# Optimizer and Training Hyperparameters
###############################
# These parameters directly impact the training process and are candidates for hyperparameter optimization.

TRAINING_CONFIG = {
    # Learning rate for the optimizer.
    "learning_rate": 0.001,  # A critical hyperparameter; too high can cause divergence, too low slows convergence.

    # Number of epochs for training.
    "num_epochs": 5,  # Total passes through the training data; impacts both training time and potential overfitting.

    # Batch size used during training and validation.
    "batch_size": 50,
    # Determines how many samples are processed at once; larger values improve GPU utilization but require more memory.

    # Fraction of the dataset to use for training; the rest is used for validation.
    "train_split": 0.8,  # 80% training, 20% validation; helps gauge model generalization.

    # Random seed to ensure reproducible splits between training and validation data.
    "random_seed": 42,  # Fixing the seed ensures the same data split across different runs for fair comparisons.

    # Early stopping patience: number of epochs with no improvement on validation loss before stopping training.
    "patience": 3  # If validation loss doesn't improve for 3 consecutive epochs, training is stopped early to prevent overfitting.
}

###############################
# Data Loader Hyperparameters
###############################
# These parameters are mostly computational optimizations.

DATALOADER_CONFIG = {
    # Number of subprocesses used to load data.
    "num_workers": 10,  # More workers can speed up data loading on multi-core machines.

    # Whether to pin memory (page-locked memory) for faster transfers to the GPU.
    "pin_memory": True,  # Helps improve transfer speeds to the GPU.

    # Whether the dataset should be shuffled at each epoch.
    "shuffle": True  # Shuffling ensures batches are random, which helps the model generalize.
}

###############################
# Data Transformation Hyperparameters
###############################
# These parameters define how input images (and masks) are preprocessed.

TRANSFORM_CONFIG = {
    # Target size for resizing images and masks.
    "resize": (256, 256)
    # Standardizing image sizes; note: if aspect ratio is important, consider a different strategy.
    # This parameter is generally fixed for a given project to maintain consistency.
}

###############################
# Loss Configuration
###############################
# These settings define the loss function used and the threshold for binarizing predictions.

LOSS_CONFIG = {
    # The type of loss function used.
    "loss_type": "BCEWithLogitsLoss",  # Documentational; you will import and use nn.BCEWithLogitsLoss() in the code.

    # The threshold for converting model outputs (after sigmoid) into binary predictions.
    "threshold": 0.5  # Typically 0.5; can be tuned to adjust precision/recall trade-offs during evaluation.
}
