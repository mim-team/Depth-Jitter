# Depth-Jitter
## Overview
Depth-Jitter is an advanced image augmentation project designed to enhance datasets by incorporating depth information. This repository provides a suite of tools for **depth-aware image processing**, **data augmentation**, and **model training**, allowing researchers to improve model robustness, particularly in depth-dependent applications such as **underwater imaging, autonomous navigation, and 3D reconstruction**.

Key features of Depth-Jitter include:
- **Depth-Based Augmentation**: Modifies image intensities based on depth information to simulate real-world conditions.
- **Quantile-Based Thresholding**: Implements a data-driven thresholding technique to adaptively process images with varying depth distributions.
- **Adaptive Depth Offsetting**: Introduces controlled variations using a randomized depth jitter technique.
- **Multi-Dataset Support**: Designed to work with multiple datasets, including UTDAC2020 and FathomNet, with optimized depth thresholds.
- **Seamless Integration**: Compatible with deep learning frameworks for model training and evaluation.

By integrating depth-aware augmentation, Depth-Jitter improves model generalization and robustness, making it highly applicable to computer vision tasks in **low-visibility environments, robotics, and depth-aware object detection**.


### Underwater Image Formation Model
<p align="center">
  <img src="assets/seathru.png" alt="Alt Text" width="500"/>
</p>

### Depth Jitter 
<p align="center">
  <img src="assets/depth-jitter-white.png" alt="Alt Text" width="500"/>
</p>


## Depth Jitter Equation

To model depth-aware augmentation, we introduce the following equation:
$$
\[
\boldsymbol{I_{c_p}^{mod} = \left( I_{c_p}^{orig} - B_c (1 - e^{-\gamma_c z_p}) \right) e^{-\beta_c (\Delta z_m - z_p)} + B_c (1 - e^{-\gamma_c \Delta z_m})}
\]
$$
In this equation, \( \Delta z_m \) represents the depth offset added to the original depth map. By incorporating this offset, we generate **synthetic data with depth variations**, which serves as an effective **data augmentation strategy**. This method enhances the model’s robustness to varying **color and depth conditions**, particularly in **underwater environments** where visibility and illumination vary significantly.

By applying depth offsets during training, the model learns to generalize across different visibility settings, leading to **improved adaptability in real-world scenarios**.

## Project Structure

```
├── README.md
├── assets
│   └── project_video.mp4
├── parameters_train.json (seathru parameters)
├── q2l_labeller
│   ├── __init__.py
│   ├── __pycache__
│   ├── data (contains custom augmentation file dataset.py)
│   ├── loss_modules
│   ├── models
│   └── pl_modules
├── simple-demo.ipynb (training notebook)
├── train.json (annotaiton)
├── train_q2l.py (training script)
└── val.json (annotation)
```
## Usage
### Create Conda Environment
```
conda env create -f environment.yml

```
### Activate Conda Environment
```
conda activate depth-jitter

```
