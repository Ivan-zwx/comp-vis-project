# Introduction to Computer Vision - Project

## 1. Introduction
This project explores binary semantic segmentation using convolutional neural networks (CNNs). The objective is to detect and segment objects in images by classifying each pixel as either foreground (object) or background. This report documents the process, models, training and evaluation methodologies, and a comparison of different U-Net architectures.

## 2. Dataset Description
The dataset used in this project is the **Carvana dataset** from Kaggle (https://www.kaggle.com/competitions/carvana-image-masking-challenge/data). The dataset originally consists of high-resolution images of cars along with their corresponding segmentation masks. This project uses:
- **Training Data:** Images and corresponding masks from the `train` and `train_masks` directories.
- **Test Data:** A separate test set without masks, from which a subset was manually labeled using Roboflow for evaluation purposes.

> **Dataset Visualization**
![Visualization](https://github.com/Ivan-zwx/comp-vis-project/blob/master/other/simple_visualization.png "Visualization")

## 3. Model Architecture Overview
All models in this project are based on the U-Net architecture, which is well-suited for semantic segmentation due to its encoder–decoder structure and skip connections that help in preserving spatial details.

> **U-Net Diagram:**
![U-Net](https://github.com/Ivan-zwx/comp-vis-project/blob/master/other/u-net_diagram.png "U-Net")

## 4. Implemented Models
Three variants of U-Net were implemented in this project:

1. **U-Net with ResNet-34 Encoder:**  
   - Uses a pre-trained ResNet-34 backbone as the encoder.
   - Uses imagenet initial model weights.
   - The decoder is automatically adapted by the segmentation models library to match the encoder features.
   
2. **U-Net with EfficientNet-B0 Encoder:**
   - Similar to the first model but replaces the ResNet-34 backbone with an EfficientNet-B0 backbone.
   - The EfficientNet backbone provides different feature representations, which may affect performance.
   
3. **Custom U-Net:**  
   - A fully handcrafted U-Net architecture, with both encoder and decoder defined from scratch.
   - Designed for simplicity and clarity, this model does not rely on pre-trained weights and is implemented entirely in PyTorch.

## 5. Training and Evaluation Methodology
- **Dataset:**  
  The original training data is split into a training set and a validation set using a fixed random seed (to ensure reproducibility).
  
- **Hyperparameters:**  
  Consistent hyperparameters (learning rate, batch size, number of epochs, etc.) are used across all models as defined in the configuration file.
  
- **Metrics:**  
  - Training Loss and Validation Loss
  - Dice Coefficient and Intersection over Union (IoU)
  
- **Early Stopping and Checkpointing:**  
  The training is monitored with early stopping (based on validation loss) to prevent overfitting. The best model checkpoint is saved and used for final evaluation.

## 6. Model Training Performance

### 6.1. U-Net with ResNet-34 Encoder
> **Training/Validation Loss Curves | Dice and IoU Curves:**  
  ![Model_1](https://github.com/Ivan-zwx/comp-vis-project/blob/master/results/resnet34/resnet34_3_training_plot.png "Model_1")

**Metrics:**  
- Epoch: 14 (Checkpoint)
- Train Loss: 0.005553
- Val Loss: 0.007029
- Val Dice: 0.993720
- Val IoU: 0.987526

### 6.2. U-Net with EfficientNet-B0 Encoder
> **Training/Validation Loss Curves | Dice and IoU Curves:**  
  ![Model_2](https://github.com/Ivan-zwx/comp-vis-project/blob/master/results/efficientnet-b0/efficientnet-b0_1_training_plot.png "Model_2")

**Metrics:**  
- Epoch: 20 (End)
- Train Loss: 0.004582
- Val Loss: 0.006240
- Val Dice: 0.994469
- Val IoU: 0.989003

### 6.3. Custom U-Net Architecture
> **Training/Validation Loss Curves | Dice and IoU Curves:**  
  ![Model_3](https://github.com/Ivan-zwx/comp-vis-project/blob/master/results/custom_unet/custom_unet_2_training_plot.png "Model_3")

**Metrics:**  
- Epoch: 20 (End)
- Train Loss: 0.007243
- Val Loss: 0.007670
- Val Dice: 0.993153
- Val IoU: 0.986409

## 7. Data Enrichment and Manual Annotation
To further evaluate the models, a subset of the test dataset (which was originally unlabeled) was manually annotated using Roboflow (https://app.roboflow.com/orv-exercises/orv-carvana). Basic data augmentation techniques were applied programmatically to enhance this manually labeled subset.

## 8. Test Performance on Manually Labeled Data

**U-Net with ResNet-34 Encoder**

**U-Net with EfficientNet-B0 Encoder**

**Custom U-Net**

## 9. Conclusion
This project demonstrated the implementation of binary semantic segmentation using U-Net architectures with different backbones and a fully custom U-Net. While the primary goal was not to maximize performance, each model provided insights into segmentation performance and model complexity.
- A robust training process was implemented.
- Different backbones can significantly affect training dynamics and performance.
- The custom U-Net provides a clear, straightforward baseline that highlights the core principles of encoder–decoder architectures.
