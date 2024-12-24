# Depth-Jitter

<video width="320" height="240" controls>
  <source src="/Users/mdsazidurrahman/Depth-Jitter/assets/project_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Overview

Depth-Jitter is a project focused on augmenting image datasets with depth information. This repository contains various modules for data processing, augmentation, and model training.

## Project Structure

'''
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
'''
## Usage