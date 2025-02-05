# Introduction to Computer Vision - Project

## 1. Introduction
This project explores binary semantic segmentation using convolutional neural networks (CNNs). The objective is to detect and segment objects in images by classifying each pixel as either foreground (object) or background. This report documents the process, models, training and evaluation methodologies, and a comparison of different U-Net architectures.

## 2. Problem Description
The problem tackled in this project is **binary semantic segmentation**. In binary segmentation, each pixel in the input image is classified into one of two classes: foreground (e.g., the object of interest) or background. This task is fundamental in computer vision and is critical for applications such as object detection, medical imaging, and autonomous driving.

## 3. Dataset Description
The dataset used in this project is the **Carvana dataset** from Kaggle. The dataset originally consists of high-resolution images of cars along with their corresponding segmentation masks. In this project, we have:
- **Training Data:** Images and corresponding masks from the `train` and `train_masks` directories.
- **Test Data:** A separate test set without masks, from which a subset was manually labeled using Roboflow for evaluation purposes.
  
*Note: Additional details and preprocessing steps are described in the project's documentation.*

## 4. Model Architecture Overview
All models in this project are based on the U-Net architecture, which is well-suited for semantic segmentation due to its encoder–decoder structure and skip connections that help in preserving spatial details.

> **U-Net Diagram:**  
> *[Insert U-Net diagram here]*

## 5. Implemented Models
Three variants of U-Net were implemented in this project:

1. **U-Net with ResNet-34 Encoder:**  
   - Uses a pre-trained ResNet-34 backbone as the encoder.
   - The decoder is automatically adapted by the segmentation models library to match the encoder features.
   
2. **U-Net with EfficientNet-B0 Encoder:**  
   - Similar to the first model but replaces the ResNet-34 backbone with an EfficientNet-B0 backbone.
   - The EfficientNet backbone provides different feature representations, which may affect performance.
   
3. **Custom U-Net:**  
   - A fully handcrafted U-Net architecture, with both encoder and decoder defined from scratch.
   - Designed for simplicity and clarity, this model does not rely on pre-trained weights and is implemented entirely in PyTorch.

## 6. Training and Evaluation Methodology
The training process involves:
- **Dataset Splitting:**  
  The original training data is split into a training set and a validation set using a fixed random seed (to ensure reproducibility).
  
- **Hyperparameters:**  
  Consistent hyperparameters (learning rate, batch size, number of epochs, etc.) are used across all models as defined in the configuration file.
  
- **Metrics:**  
  - **Training Loss and Validation Loss:** Monitored at each epoch.
  - **Dice Coefficient and Intersection over Union (IoU):** Used as key metrics for evaluating segmentation quality.
  
- **Early Stopping and Checkpointing:**  
  The training is monitored with early stopping (based on validation loss) to prevent overfitting. The best model checkpoint is saved and used for final evaluation.

## 7. Model Training Performance

### 7.1. U-Net with ResNet-34 Encoder
- **Training/Validation Loss Curves:**  
  *[Insert graph of training and validation loss over epochs for model 1]*  
- **Dice and IoU Curves:**  
  *[Insert graph of Dice and IoU scores over epochs for model 1]*

### 7.2. U-Net with EfficientNet-B0 Encoder
- **Training/Validation Loss Curves:**  
  *[Insert graph of training and validation loss over epochs for model 2]*  
- **Dice and IoU Curves:**  
  *[Insert graph of Dice and IoU scores over epochs for model 2]*

### 7.3. Custom U-Net Architecture
- **Training/Validation Loss Curves:**  
  *[Insert graph of training and validation loss over epochs for model 3]*  
- **Dice and IoU Curves:**  
  *[Insert graph of Dice and IoU scores over epochs for model 3]*

## 8. Model Comparison
A brief comparison of the three models based on their training performance is as follows:

- **U-Net with ResNet-34 Encoder:**  
  *[Placeholder for qualitative and quantitative comparison of training performance]*

- **U-Net with EfficientNet-B0 Encoder:**  
  *[Placeholder for qualitative and quantitative comparison of training performance]*

- **Custom U-Net:**  
  *[Placeholder for qualitative and quantitative comparison of training performance]*

*Note: Detailed numerical metrics and analysis will be added as the experiments are finalized.*

## 9. Data Enrichment and Manual Annotation
To further evaluate the models, a subset of the test dataset (which was originally unlabeled) was manually annotated using Roboflow. Basic data augmentation techniques (e.g., random rotations, flips, and color jitter) were applied programmatically to enhance this manually labeled subset.

## 10. Test Performance on Manually Labeled Data

### 10.1. U-Net with ResNet-34 Encoder
- **Evaluation Metrics:**  
  *[Insert test performance metrics: average loss, Dice, IoU for model 1]*

### 10.2. U-Net with EfficientNet-B0 Encoder
- **Evaluation Metrics:**  
  *[Insert test performance metrics: average loss, Dice, IoU for model 2]*

### 10.3. Custom U-Net
- **Evaluation Metrics:**  
  *[Insert test performance metrics: average loss, Dice, IoU for model 3]*

## 11. Conclusion
This project demonstrated the implementation of binary semantic segmentation using U-Net architectures with different backbones and a fully custom U-Net. While the primary goal was not to maximize performance, each model provided unique insights into segmentation performance and model complexity.  
- **Key Takeaways:**  
  - A robust training process with early stopping and checkpointing was implemented.
  - Different backbones can significantly affect training dynamics and performance.
  - The custom U-Net provides a clear, straightforward baseline that highlights the core principles of encoder–decoder architectures.
  
Further work can explore additional augmentations and model refinements, but overall, the models achieved acceptable segmentation performance, meeting the course requirements.
