img_data_dir = "/home/mundus/mrahman528/thesis/detr/dataset/"

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger # Comment out if not using wandb
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import StochasticWeightAveraging
import torch
# torch.multiprocessing.set_start_method('spawn', force=True)
from q2l_labeller.data.coco_data_module import COCODataModule
from q2l_labeller.pl_modules.query2label_train_module import Query2LabelTrainModule
from q2l_labeller.data.dataset import SeaThruAugmentation
pl.seed_everything(42)

param_dict = {
    "backbone_desc":"resnest101e",
    "conv_out_dim":2048,
    "hidden_dim":256,
    "num_encoders":1,
    "num_decoders":2,
    "num_heads":8,
    "batch_size":64,
    "image_dim":384,
    "learning_rate":0.0001,
    "momentum":0.9,
    "weight_decay":0.01,
    "n_classes":290,
    "thresh":0.5,
    "use_cutmix":True,
    "use_pos_encoding":True,
    "loss":"ASL",
    "use_seathru": True  # Add this line to enable or disable SeaThru augmentation
}

train_classes = [160, 51, 119, 37, 52, 10, 88, 146, 125, 1, 260, 133, 9, 214, 70, 120, 111, 142, 274, 105, 69, 174, 203, 103, 228, 259, 205, 104, 116, 242, 16, 219, 81, 61, 100, 11, 224, 202, 82, 108, 255, 3, 54, 162, 85, 256, 8, 67, 71, 75, 173, 201, 93, 243, 218, 131, 99, 43, 36, 283]

image_folder = '/home/mundus/mrahman528/thesis/query2label/train/'
depth_image_folder = '/home/mundus/mrahman528/thesis/Depth-Anything-V2/depth_vis_train'
depth_npy_folder = '/home/mundus/mrahman528/thesis/Depth-Anything-V2/depth_vis_train'
seathru_parameters_path = '/home/mundus/mrahman528/thesis/sucre/src/output/parameters_train.json'
seathru_transform = SeaThruAugmentation(image_folder, depth_image_folder, depth_npy_folder, seathru_parameters_path)

coco = COCODataModule(
    data_dir="/home/mundus/mrahman528/thesis/query2label/",
    img_size=384,
    batch_size=128,
    num_workers=0,
    use_cutmix=False,
    cutmix_alpha=1.0,
    train_classes=None,
    use_seathru=True,
    seathru_transform=seathru_transform
)

param_dict["data"] = coco

pl_model = Query2LabelTrainModule(**param_dict)

# Comment out if not using wandb
# wandb_logger = WandbLogger(
#     project="fathomnet_osd", 
#     save_dir="training/logs/fathomnet_with_all_seathru",
#     log_model=True)
# wandb_logger.watch(pl_model, log="all")

trainer = pl.Trainer(
    max_epochs=10,
    precision=16,
    accelerator='gpu', 
    devices=1,  # Use all available GPUs
    # strategy='ddp',  # Use Distributed Data Parallel strategy
    # logger=wandb_logger, # Comment out if not using wandb
    default_root_dir="training/checkpoints/depth_jitter",
    callbacks=[TQDMProgressBar(refresh_rate=10)]
)
trainer.fit(pl_model, param_dict["data"])
