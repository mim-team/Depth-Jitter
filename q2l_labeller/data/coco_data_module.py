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
        augmentation_strategy="traditional",  # traditional or seathru
        seathru_transform=None
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_cutmix = use_cutmix
        self.cutmix_alpha = cutmix_alpha
        self.collator = torch.utils.data.dataloader.default_collate
        self.train_classes = train_classes
        self.augmentation_strategy = augmentation_strategy
        self.seathru_transform = seathru_transform if augmentation_strategy == "seathru" else None


    def prepare_data(self) -> None:
        """Loads metadata file and subsamples it if requested"""
        pass

    def setup(self, stage=None) -> None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.augmentation_strategy == "seathru" and self.seathru_transform:
            train_transforms = self.seathru_transform
        else:
            train_transforms = transforms.Compose([
                RandAugment(),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                normalize,
            ])
        test_transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            normalize,
        ])

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
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
            persistent_workers=True,  # Keep workers alive between epochs
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
            pin_memory=True,
            shuffle=False,
            persistent_workers=True
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