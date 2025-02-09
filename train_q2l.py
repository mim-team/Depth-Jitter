import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    TQDMProgressBar, ModelCheckpoint, EarlyStopping
)
from q2l_labeller.data.coco_data_module import COCODataModule
from q2l_labeller.pl_modules.query2label_train_module import Query2LabelTrainModule
from q2l_labeller.data.dataset import SeaThruAugmentation

# Set random seed
pl.seed_everything(40)
torch.backends.cudnn.benchmark = True

# Argument Parser for Dynamic Dataset Selection
parser = argparse.ArgumentParser(description="Depth-Jitter Training Script")
parser.add_argument(
    "--dataset", type=str, choices=["UTDAC2020", "FathomNet"], default="FathomNet",
    help="Select dataset: 'UTDAC2020' or 'FathomNet' (default: FathomNet)"
)
args = parser.parse_args()

# Dataset configurations
datasets = {
    "UTDAC2020": {
        "image_folder": "/home/mundus/mrahman528/thesis/thesis_paper/UTDAC2020",
        "depth_image_folder": "/home/mundus/mrahman528/thesis/thesis_paper/UTDAC2020/depth_train",
        "depth_npy_folder": "/home/mundus/mrahman528/thesis/thesis_paper/UTDAC2020/depth_train",
        "seathru_parameters_path": "/home/mundus/mrahman528/thesis/thesis_paper/parameters_train.json",
        "depth_variance_path": "/home/mundus/mrahman528/Depth-Jitter/depth_variance_utdac.json",
        "threshold": 9.49,  # Precomputed threshold
        "num_classes": 4
    },
    "FathomNet": {
        "image_folder": "/home/mundus/mrahman528/projects/mir/depth_jitter/fathomnet_2023_dataset",
        "depth_image_folder": "/home/mundus/mrahman528/projects/mir/depth_jitter/fathomnet_2023_dataset/depth_vis_train",
        "depth_npy_folder": "/home/mundus/mrahman528/projects/mir/depth_jitter/fathomnet_2023_dataset/depth_vis_train",
        "seathru_parameters_path": "/home/mundus/mrahman528/Depth-Jitter/parameters_train.json",
        "depth_variance_path": "/home/mundus/mrahman528/Depth-Jitter/depth_variance_fathomnet.json",
        "threshold": 3.66,  # Precomputed threshold
        "num_classes": 290
    }
}

# Select dataset based on user input
selected_dataset = datasets[args.dataset]

# Initialize SeaThru Augmentation
seathru_transform = SeaThruAugmentation(
    selected_dataset["image_folder"],
    selected_dataset["depth_image_folder"],
    selected_dataset["depth_npy_folder"],
    selected_dataset["seathru_parameters_path"],
    selected_dataset["depth_variance_path"],
    threshold=selected_dataset["threshold"]
)

# Initialize Data Module
coco = COCODataModule(
    data_dir=selected_dataset["image_folder"],
    img_size=384,
    batch_size=128,
    num_workers=8,  # Adjust based on CPU cores
    use_cutmix=True,
    cutmix_alpha=1.0,
    train_classes=None,
    sampling_strategy="oversample",  # oversample, undersample, default
    augmentation_strategy="seathru",
    num_classes=selected_dataset["num_classes"],
    seathru_transform=seathru_transform
)

# Model Parameters (Updated n_classes Dynamically)
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

# Initialize Model
pl_model = Query2LabelTrainModule(**param_dict)

# WandB Logger
wandb_logger = WandbLogger(
    project="depth_jitter-last-final",
    save_dir="training/logs/depthJitter",
    log_model=True,
    id=f"resnest-DJ+all-ASL-UTDAC-384",  # Unique experiment ID
    sync_tensorboard=True
)

# Model Checkpoint Callback
checkpoint_callback = ModelCheckpoint(
    monitor="val_mAP",
    dirpath=f"training/checkpoints/depth_jitter_{args.dataset}",
    filename="best-checkpoint-{epoch:02d}-{val_mAP:.2f}",
    save_top_k=1,
    mode="min"
)

# Early Stopping Callback
early_stopping_callback = EarlyStopping(
    monitor="val_mAP",
    patience=30,  # Number of epochs with no improvement
    verbose=True,
    mode="min"
)

# Trainer Configuration
trainer = pl.Trainer(
    max_epochs=200,
    precision=16,
    accelerator="gpu",
    devices="auto",
    strategy="ddp",
    gradient_clip_val=0.1,
    logger=wandb_logger,
    default_root_dir=f"training/checkpoints/depth_jitter_{args.dataset}",
    callbacks=[
        TQDMProgressBar(refresh_rate=100),
        checkpoint_callback,
        early_stopping_callback
    ],
    accumulate_grad_batches=4,
    detect_anomaly=True,
    profiler="simple"
)

# Start Training
trainer.fit(pl_model, param_dict["data"])
