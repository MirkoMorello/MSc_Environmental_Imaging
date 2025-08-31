# Semantic Segmentation of High-Resolution Aerial Drone Imagery

> **Course:** Signal and Imaging Acquisition and Modelling in Environment

## Overview

This project addresses the significant challenge of performing semantic segmentation on extremely high-resolution aerial images (6000x4000 pixels) captured by drones. Due to computational and memory constraints, processing such large images at once is infeasible. This work develops and evaluates a robust pipeline based on a patch-based approach, incorporating multi-scale analysis to ensure both local detail accuracy and global contextual understanding.

## Methodology

### 1. Patch-Based Processing
Images are divided into smaller, overlapping patches (e.g., 1000x1000 pixels) that can be processed efficiently by a GPU.

### 2. Multi-Scale "Stitch Level" Approach
To help the model understand context beyond a single patch, we introduced a novel multi-scale stitching method. This involves generating patches at multiple zoom levels (e.g., 1x, 2x, 4x) and resizing them to a uniform input size. This technique allows the model to learn features at various spatial resolutions simultaneously.

### 3. Models and Architectures
We implemented and compared several deep learning models for this task:
*   **DeepLabV3+** with **ResNet34** and **EfficientNet-B5** backbones (leveraging pre-trained ImageNet weights).
*   A **custom-designed model** inspired by U-Net and DeepLab architectures, featuring an ASPP module and attention blocks.

### 4. Training and Optimization
*   **Data Augmentation:** An extensive augmentation pipeline using `Albumentations` was employed to improve model generalization.
*   **Memory Management:** To handle the large models and patch sizes, we implemented **gradient accumulation**, which allows for training with larger effective batch sizes than would otherwise fit in GPU memory.

## Technologies Used

*   **Deep Learning:** Python, PyTorch, `timm`
*   **Data Augmentation:** Albumentations
*   **Image Processing:** OpenCV, Pillow, NumPy
