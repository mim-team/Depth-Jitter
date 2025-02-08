# Depth-Jitter

### Underwater Image Formation Model
<p align="center">
  <img src="assets/seathru.png" alt="Alt Text" width="500"/>
</p>

### Depth Jitter 
<p align="center">
  <img src="assets/depth-jitter-white.png" alt="Alt Text" width="500"/>
</p>


## Overview

Depth-Jitter is a project focused on augmenting image datasets with depth information. This repository contains various modules for data processing, augmentation, and model training.

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
