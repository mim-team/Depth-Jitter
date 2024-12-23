# import os
# import torch
# import pandas as pd
# import pytorch_lightning as pl
# import torchvision.transforms as transforms
# from randaugment import RandAugment
# from torch.utils.data import DataLoader
# from q2l_labeller.data.coco_dataset import CoCoDataset
# from q2l_labeller.data.cutmix import CutMixCollator
# # from q2l_labeller.data.dataset import SeaThru
# from q2l_labeller.data.dataset import SeaThruAugmentation

# # Initialize SeaThruAugmentation
# image_folder = '/home/mundus/mrahman528/thesis/query2label/train/'
# depth_image_folder = '/home/mundus/mrahman528/thesis/Depth-Anything-V2/depth_vis_train'
# depth_npy_folder = '/home/mundus/mrahman528/thesis/Depth-Anything-V2/depth_vis_train'
# seathru_parameters_path = '/home/mundus/mrahman528/thesis/sucre/src/output/parameters_train.json'

# class COCODataModule(pl.LightningDataModule):
#     """Datamodule for Lightning Trainer"""

#     def __init__(
#         self,
#         data_dir,
#         img_size,
#         batch_size=128,
#         num_workers=0,
#         use_cutmix=False,
#         cutmix_alpha=1.0,
#         train_classes=None,
#         use_seathru=True,
#         seathru_transform=None
#     ) -> None:
#         """_summary_

#         Args:
#             data_dir (str): Location of data.
#             img_size (int): Desired size for transformed images.
#             batch_size (int, optional): Dataloader batch size. Defaults to 128.
#             num_workers (int, optional): Number of CPU threads to use. Defaults to 0.
#             use_cutmix (bool, optional): Flag to enable Cutmix augmentation. Defaults to False.
#             cutmix_alpha (float, optional): Defaults to 1.0.
#             use_seathru (bool, optional): Flag to enable SeaThru augmentation. Defaults to False.
#         """
#         super().__init__()
#         self.data_dir = data_dir
#         self.img_size = img_size
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.use_cutmix = use_cutmix
#         self.cutmix_alpha = cutmix_alpha
#         self.collator = torch.utils.data.dataloader.default_collate
#         self.train_classes = train_classes
#         self.use_seathru = use_seathru
#         self.seathru_transform = seathru_transform

#     def prepare_data(self) -> None:
#         """Loads metadata file and subsamples it if requested"""
#         pass

#     def setup(self, stage=None) -> None:
#         """Creates train, validation, test datasets
        
#         Args:
#             stage (str, optional): Stage. Defaults to None.
#         """
        
#         normalize = transforms.Normalize(
#             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#         )

#         train_transforms = transforms.Compose(
#             [
#                 # transforms.Resize((self.img_size, self.img_size)),
#                 # RandAugment(),
#                 # transforms.ToTensor(),
#                 normalize,
#             ]
#         )

#         test_transforms = transforms.Compose(
#             [
#                 # transforms.Resize((self.img_size, self.img_size)),
#                 # transforms.ToTensor(),
#                 normalize,
#             ]
#         )
#         seathru_transform = SeaThruAugmentation(image_folder, depth_image_folder, depth_npy_folder, seathru_parameters_path)
#         # Define train set
#         if stage == 'fit' or stage is None:
#             self.train_set = CoCoDataset(
#                 image_dir="/home/mundus/mrahman528/thesis/query2label/train/",
#                 anno_path="/home/mundus/mrahman528/thesis/query2label/train.json",
#                 input_transform=train_transforms,
#                 labels_path=(self.data_dir + "annotations/train.npy"),
#                 train_classes=self.train_classes,
#                 seathru_transform=self.seathru_transform if self.use_seathru else None  # Pass the transform here if enabled
#                 # seathru_transform = seathru_transform
#             )
        
#         # Define validation set
#         self.val_set = CoCoDataset(
#                 image_dir="/home/mundus/mrahman528/thesis/query2label/val",
#                 anno_path="/home/mundus/mrahman528/thesis/query2label/val.json",
#                 input_transform=test_transforms,
#                 labels_path=(self.data_dir + "annotations/val.npy"),
#                 train_classes=self.train_classes,
#                 # seathru_transform=self.seathru_transform if self.use_seathru else None  # Pass the transform here if enabled
#                 # seathru_transform = seathru_transform
#             )

#         # Optional: Define test set
#         if stage == 'test' or stage is None:
#             self.test_set = CoCoDataset(
#                 image_dir=("/home/mundus/mrahman528/thesis/fgvc-comp-2023/eval_images"),
#                 anno_path=("/home/mundus/mrahman528/thesis/fgvc-comp-2023/object_detection/eval.json"),
#                 input_transform=test_transforms,
#                 labels_path=("/home/mundus/mrahman528/thesis/query2label/test.npy")
#             )

#         # Update collator if using cutmix
#         if self.use_cutmix:
#             self.collator = CutMixCollator(self.cutmix_alpha)

#         # Optional: Filter samples based on train classes
#         if self.train_classes is not None:
#             self.train_set.filter_samples(self.train_classes)

#     def get_num_classes(self):
#         """Returns number of classes

#         Returns:
#             int: number of classes
#         """
#         return len(self.classes)

#     def train_dataloader(self) -> DataLoader:
#         """Creates and returns training dataloader

#         Returns:
#             DataLoader: Training dataloader
#         """
#         return DataLoader(
#             self.train_set,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=True,
#             collate_fn=self.collator,
#         )

#     def val_dataloader(self) -> DataLoader:
#         """Creates and returns validation dataloader

#         Returns:
#             DataLoader: Validation dataloader
#         """
#         return DataLoader(
#             self.val_set,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=False,
#         )

#     def test_dataloader(self) -> DataLoader:
#         """Creates and returns test dataloader

#         Returns:
#             DataLoader: Test dataloader
#         """
#         return DataLoader(
#             self.test_set,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=False,
#         )
import os
import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
from randaugment import RandAugment
from torch.utils.data import DataLoader
from q2l_labeller.data.coco_dataset import CoCoDataset
from q2l_labeller.data.cutmix import CutMixCollator
from q2l_labeller.data.dataset import SeaThruAugmentation

class COCODataModule(pl.LightningDataModule):
    """Datamodule for Lightning Trainer"""

    def __init__(
        self,
        data_dir,
        img_size,
        batch_size=128,
        num_workers=0,
        use_cutmix=False,
        cutmix_alpha=1.0,
        train_classes=None,
        use_seathru=False,
        seathru_transform=None
    ) -> None:
        """_summary_

        Args:
            data_dir (str): Location of data.
            img_size (int): Desired size for transformed images.
            batch_size (int, optional): Dataloader batch size. Defaults to 128.
            num_workers (int, optional): Number of CPU threads to use. Defaults to 0.
            use_cutmix (bool, optional): Flag to enable Cutmix augmentation. Defaults to False.
            cutmix_alpha (float, optional): Defaults to 1.0.
            use_seathru (bool, optional): Flag to enable SeaThru augmentation. Defaults to False.
        """
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_cutmix = use_cutmix
        self.cutmix_alpha = cutmix_alpha
        self.collator = torch.utils.data.dataloader.default_collate
        self.train_classes = train_classes
        self.use_seathru = use_seathru
        self.seathru_transform = seathru_transform if use_seathru else None


    def prepare_data(self) -> None:
        """Loads metadata file and subsamples it if requested"""
        pass

    def setup(self, stage=None) -> None:
        """Creates train, validation, test datasets
        
        Args:
            stage (str, optional): Stage. Defaults to None.
        """
        
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        common_transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            normalize,
        ])

        train_transforms = transforms.Compose([
            RandAugment(),
            common_transforms,
        ])

        test_transforms = common_transforms

        # Define train set
        if stage == 'fit' or stage is None:
            self.train_set = CoCoDataset(
                image_dir=os.path.join(self.data_dir, "train"),
                anno_path=os.path.join(self.data_dir, "train.json"),
                input_transform=train_transforms,
                labels_path=os.path.join(self.data_dir, "annotations/train.npy"),
                train_classes=self.train_classes,
                seathru_transform=self.seathru_transform
            )
        
        # Define validation set
        self.val_set = CoCoDataset(
            image_dir=os.path.join(self.data_dir, "val"),
            anno_path=os.path.join(self.data_dir, "val.json"),
            input_transform=test_transforms,
            labels_path=os.path.join(self.data_dir, "annotations/val.npy"),
            train_classes=self.train_classes,
            seathru_transform=self.seathru_transform
        )

        # Optional: Define test set
        if stage == 'test' or stage is None:
            self.test_set = CoCoDataset(
                image_dir=os.path.join(self.data_dir, "eval_images"),
                anno_path=os.path.join(self.data_dir, "object_detection/eval.json"),
                input_transform=test_transforms,
                labels_path=os.path.join(self.data_dir, "annotations/test.npy")
            )

        # Update collator if using cutmix
        if self.use_cutmix:
            self.collator = CutMixCollator(self.cutmix_alpha)

        # Optional: Filter samples based on train classes
        if self.train_classes is not None:
            self.train_set.filter_samples(self.train_classes)

    def get_num_classes(self):
        """Returns number of classes

        Returns:
            int: number of classes
        """
        return len(self.classes)

    def train_dataloader(self) -> DataLoader:
        """Creates and returns training dataloader

        Returns:
            DataLoader: Training dataloader
        """
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collator,
        )

    def val_dataloader(self) -> DataLoader:
        """Creates and returns validation dataloader

        Returns:
            DataLoader: Validation dataloader
        """
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Creates and returns test dataloader

        Returns:
            DataLoader: Test dataloader
        """
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
