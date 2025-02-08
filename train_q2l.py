import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger # Comment out if not using wandb
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import StochasticWeightAveraging
import torch
# torch.multiprocessing.set_start_method('spawn', force=True)
from q2l_labeller.data.coco_data_module import COCODataModule
from q2l_labeller.pl_modules.query2label_train_module import Query2LabelTrainModule
from q2l_labeller.data.dataset import SeaThruAugmentation
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
pl.seed_everything(40)
torch.backends.cudnn.benchmark = True
import os 

# log_dir = "training/logs/depthJitter"
# os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist
param_dict = {
    "backbone_desc":"resnest101e",
    "conv_out_dim":2048,
    "hidden_dim":256,
    "num_encoders":2,
    "num_decoders":3,
    "num_heads":8,
    "batch_size":128,
    "image_dim":384,
    "learning_rate":1e-4,
    "momentum":0.9,
    "weight_decay":1e-2,
    "n_classes":290,
    "thresh":0.4,
    "use_cutmix":True,
    "use_pos_encoding":True,
    "loss":"ASL",
}

# train_classes = [160, 51, 119, 37, 52, 10, 88, 146, 125, 1, 260, 133, 9, 214, 70, 120, 111, 142, 274, 105, 69, 174, 203, 103, 228, 259, 205, 104, 116, 242, 16, 219, 81, 61, 100, 11, 224, 202, 82, 108, 255, 3, 54, 162, 85, 256, 8, 67, 71, 75, 173, 201, 93, 243, 218, 131, 99, 43, 36, 283]

image_folder = '/home/mundus/mrahman528/projects/mir/depth_jitter/fathomnet_2023_dataset'
depth_image_folder = '/home/mundus/mrahman528/projects/mir/depth_jitter/fathomnet_2023_dataset/depth_vis_train'
depth_npy_folder = '/home/mundus/mrahman528/projects/mir/depth_jitter/fathomnet_2023_dataset/depth_vis_train'
seathru_parameters_path = '/home/mundus/mrahman528/Depth-Jitter/parameters_train.json'
depth_variance_path = "/home/mundus/mrahman528/Depth-Jitter/depth_variance_fathomnet.json"
# image_folder = "/home/mundus/mrahman528/thesis/thesis_paper/UTDAC2020"
# depth_image_folder = "/home/mundus/mrahman528/thesis/thesis_paper/UTDAC2020/depth_train"
# depth_npy_folder = "/home/mundus/mrahman528/thesis/thesis_paper/UTDAC2020/depth_train"
# seathru_parameters_path = "/home/mundus/mrahman528/thesis/thesis_paper/parameters_train.json"

seathru_transform = SeaThruAugmentation(image_folder, depth_image_folder, depth_npy_folder, seathru_parameters_path,depth_variance_path, threshold=7.5)

coco = COCODataModule(
    data_dir="/home/mundus/mrahman528/projects/mir/depth_jitter/fathomnet_2023_dataset",
    img_size=384,
    batch_size=128,
    num_workers=8,  # Adjust based on CPU cores
    use_cutmix=True,
    cutmix_alpha=1.0,
    train_classes=None,
    sampling_strategy="oversample", # oversample, undersample, default
    augmentation_strategy="seathru",
    seathru_transform=seathru_transform
)

param_dict["data"] = coco

pl_model = Query2LabelTrainModule(**param_dict)

# Wandb Logger
wandb_logger = WandbLogger(
    project="depth_jitter-last-final",
    save_dir="training/logs/depthJitter",
    log_model=True,
    id="resnest-DJ+all-ASL-Fathomnet-384",  # Unique experiment ID
    sync_tensorboard=True  # Sync logs across GPUs
)

# Model Checkpoint Callback
checkpoint_callback = ModelCheckpoint(
    monitor="val_mAP",
    dirpath="training/checkpoints/depth_jitter",
    filename="best-checkpoint-{epoch:02d}-{val_mAP:.2f}",
    save_top_k=1,
    mode="min"
)
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
    accelerator='gpu',
    devices='auto',
    strategy='ddp',
    gradient_clip_val=0.1,
    logger=wandb_logger,
    default_root_dir="training/checkpoints/depth_jitter",
    callbacks=[
        TQDMProgressBar(refresh_rate=100),
        checkpoint_callback
    ],
    accumulate_grad_batches=4,
    detect_anomaly=True,
    profiler="simple"  # Use advanced profiling if needed
)

# Start Training
trainer.fit(pl_model, param_dict["data"])

