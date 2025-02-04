import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        """
        A simple U-Net architecture for binary segmentation.

        Args:
            in_channels (int): Number of channels in the input image (e.g., 3 for RGB).
            out_channels (int): Number of output channels (e.g., 1 for binary segmentation).

        Architecture Overview:
            - Encoder: Four downsampling stages. Each stage applies two convolutions (with ReLU)
              then downsamples the feature map using max pooling.
            - Bottleneck: Two convolutions at the smallest resolution.
            - Decoder: Three upsampling stages. Each stage upsamples the feature map, concatenates
              with the corresponding encoder output (skip connection), and applies two convolutions.
            - Final Layer: A 1x1 convolution maps features to the output segmentation mask.
        """
        super(CustomUNet, self).__init__()

        # -------------------------------
        # Encoder (Contracting Path)
        # -------------------------------
        # Stage 1:
        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)  # Convolution: extracts low-level features.
        self.relu1_1 = nn.ReLU(inplace=True)  # Activation: introduces non-linearity.
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # Second convolution in stage 1.
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample: reduces spatial dimensions by half.

        # Stage 2:
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 3:
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # -------------------------------
        # Bottleneck
        # -------------------------------
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)

        # -------------------------------
        # Decoder (Expanding Path)
        # -------------------------------
        # Stage 1 (upsample and concatenate with encoder stage 3):
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # Upsamples from 512 channels to 256.
        # After concatenation, channels double (256 from upsample + 256 from skip connection)
        self.conv5_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)

        # Stage 2 (upsample and concatenate with encoder stage 2):
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # 128 (upsampled) + 128 (skip)
        self.relu6_1 = nn.ReLU(inplace=True)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu6_2 = nn.ReLU(inplace=True)

        # Stage 3 (upsample and concatenate with encoder stage 1):
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # 64 (upsampled) + 64 (skip)
        self.relu7_1 = nn.ReLU(inplace=True)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu7_2 = nn.ReLU(inplace=True)

        # -------------------------------
        # Final Convolution
        # -------------------------------
        # A 1x1 convolution maps the 64-channel output to the desired number of output channels.
        self.conv8 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # -------------------------------
        # Encoder: Stage 1
        # -------------------------------
        x1 = self.relu1_1(self.conv1_1(x))  # First convolution on input.
        x1 = self.relu1_2(self.conv1_2(x1))  # Second convolution, refining features.
        p1 = self.pool1(x1)  # Downsample by a factor of 2.

        # -------------------------------
        # Encoder: Stage 2
        # -------------------------------
        x2 = self.relu2_1(self.conv2_1(p1))
        x2 = self.relu2_2(self.conv2_2(x2))
        p2 = self.pool2(x2)

        # -------------------------------
        # Encoder: Stage 3
        # -------------------------------
        x3 = self.relu3_1(self.conv3_1(p2))
        x3 = self.relu3_2(self.conv3_2(x3))
        p3 = self.pool3(x3)

        # -------------------------------
        # Bottleneck
        # -------------------------------
        x4 = self.relu4_1(self.conv4_1(p3))
        x4 = self.relu4_2(self.conv4_2(x4))

        # -------------------------------
        # Decoder: Stage 1
        # -------------------------------
        u1 = self.upconv1(x4)  # Upsample the bottleneck output.
        # If necessary, interpolate to match the encoder output size.
        if u1.shape != x3.shape:
            u1 = F.interpolate(u1, size=x3.shape[2:])
        # Concatenate the upsampled features with the corresponding encoder output (skip connection).
        u1 = torch.cat([u1, x3], dim=1)  # Now has 512 channels (256 + 256)
        x5 = self.relu5_1(self.conv5_1(u1))  # Convolution to refine features.
        x5 = self.relu5_2(self.conv5_2(x5))

        # -------------------------------
        # Decoder: Stage 2
        # -------------------------------
        u2 = self.upconv2(x5)  # Upsample the features.
        if u2.shape != x2.shape:
            u2 = F.interpolate(u2, size=x2.shape[2:])
        u2 = torch.cat([u2, x2], dim=1)  # Concatenate with encoder stage 2 (total 256 channels).
        x6 = self.relu6_1(self.conv6_1(u2))
        x6 = self.relu6_2(self.conv6_2(x6))

        # -------------------------------
        # Decoder: Stage 3
        # -------------------------------
        u3 = self.upconv3(x6)  # Upsample the features.
        if u3.shape != x1.shape:
            u3 = F.interpolate(u3, size=x1.shape[2:])
        u3 = torch.cat([u3, x1], dim=1)  # Concatenate with encoder stage 1 (total 128 channels).
        x7 = self.relu7_1(self.conv7_1(u3))
        x7 = self.relu7_2(self.conv7_2(x7))

        # -------------------------------
        # Final Output Layer
        # -------------------------------
        output = self.conv8(x7)  # 1x1 convolution to map features to segmentation mask.
        return output
