# Depth-Jitter
<p align="center">
  <img src="assets/depthjitter.gif">
</p>

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

<p align="center">
  <img src="assets/equation.png" alt="Alt Text" width="500"/>
</p>

In this equation, \( \Delta z_m \) represents the depth offset added to the original depth map. By incorporating this offset, we generate **synthetic data with depth variations**, which serves as an effective **data augmentation strategy**. This method enhances the model’s robustness to varying **color and depth conditions**, particularly in **underwater environments** where visibility and illumination vary significantly.

By applying depth offsets during training, the model learns to generalize across different visibility settings, leading to **improved adaptability in real-world scenarios**.

## Project Structure

```
.
├── README.md
├── assets
│   ├── depth-jitter-white.png
│   ├── equation.png
│   ├── project_video.mp4
│   └── seathru.png
├── depth_variance_fathomnet.json
├── depth_variance_utdac.json
├── environment.yml
├── output-fathomnet.png
├── output-utdac.png
├── parameters_train.json
├── q2l_labeller
│   ├── __init__.py
│   ├── __pycache__
│   ├── data
│   ├── loss_modules
│   ├── models
│   └── pl_modules
├── simple-demo.ipynb
├── train.json
├── train_fathomnet.json
├── train_q2l.py
├── val.json
├── val_fathomnet.json

```
## Usage
### Clone the repository
```
git clone https://github.com/mim-team/Depth-Jitter.git
```
```
cd Depth-Jitter/
```
### Create Conda Environment
```
conda env create -f environment.yml

```
### Activate Conda Environment
```
conda activate depth-jitter

```

### train with desired dataset
```
python train_q2l.py --dataset FathomNet

```

### If you want to change augmentation settings
You can tweak the augmentation settings and the image size in this part of the training script. 
```python
# Initialize Data Module
coco = COCODataModule(
    data_dir=selected_dataset["image_folder"],
    img_size=384,
    batch_size=128,
    num_workers=8,  # Adjust based on CPU cores
    use_cutmix=True,
    cutmix_alpha=1.0,
    train_classes=None,
    sampling_strategy="default",  # oversample, undersample, default
    augmentation_strategy="seathru", # baseline, seathru, combined
    num_classes=selected_dataset["num_classes"],
    seathru_transform=seathru_transform
)
```

### Model Settings
You can change the model backbone and hyperparameters in this section of the training script. 
If you want to use different backbones you can use them from [timm](https://huggingface.co/docs/timm/en/results)

If you use a different backbone, please make sure to change the backbone_desc and conv_out_dims according to the models. 

```python

param_dict = {
    "backbone_desc": "resnest101e",
    "conv_out_dim": 2048,
    "hidden_dim": 256,
    "num_encoders": 2,
    "num_decoders": 3,
    "num_heads": 8,
    "batch_size": 128,
    "image_dim": 384,
    "learning_rate": 1e-4,
    "momentum": 0.9,
    "weight_decay": 1e-2,
    "n_classes": selected_dataset["num_classes"],  # Dynamically assign class numbers
    "thresh": 0.4,
    "use_cutmix": True,
    "use_pos_encoding": True,
    "loss": "ASL",
    "data": coco
}
```