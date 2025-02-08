import os
import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
from randaugment import RandAugment
from torch.utils.data import DataLoader
from q2l_labeller.data.coco_dataset import CoCoDataset
from q2l_labeller.data.cutmix import CutMixCollator
from q2l_labeller.data.dataset import SeaThruAugmentation
import random
from torch.utils.data import WeightedRandomSampler
import numpy as np
def compute_sample_weights(labels):
        """
        Compute sample weights for a multi-label dataset.

        Args:
            labels (numpy.ndarray): Multi-hot encoded labels for the dataset.

        Returns:
            list: List of weights for each sample.
        """
        # Calculate class frequencies
        class_counts = np.sum(labels, axis=0)  # Sum across samples for each class
        class_weights = 1.0 / np.clip(class_counts, 1, None)  # Avoid division by zero

        # Compute sample weights as the sum of class weights for each sample
        sample_weights = np.dot(labels, class_weights)
        return sample_weights.tolist()

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
        augmentation_strategy="baseline",  # traditional, seathru, combined
        seathru_transform=None,
        sampling_strategy="oversample",  # default, oversample, undersample
        combine_prob=0.5  # Probability of applying seathru in combined strategy
    ) -> None:
        super().__init__()
        self.sampling_strategy = sampling_strategy
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_cutmix = use_cutmix
        self.cutmix_alpha = cutmix_alpha
        self.collator = torch.utils.data.dataloader.default_collate
        self.train_classes = train_classes
        self.augmentation_strategy = augmentation_strategy
        self.seathru_transform = seathru_transform
        self.combine_prob = combine_prob

    def prepare_data(self) -> None:
        """Loads metadata file and subsamples it if requested"""
        pass

    
    def setup(self, stage=None) -> None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.augmentation_strategy == "seathru" and self.seathru_transform:
            print("-----------------------------DEPTHJITTER AUGMENTATION INITIALIZED-------------------------------------")
            # train_transforms = self.seathru_transform
            
            train_transforms = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),  # Resize first before augmentations
                transforms.RandomHorizontalFlip(p=0.5),  # Flipping is lightweight and should happen early
                transforms.RandomVerticalFlip(p=0.5),  # Vertical flip for more diversity
                transforms.RandomCrop(self.img_size, padding=8),  # Crop after resizing
                transforms.RandomErasing(p=0.2), # Random erasing for regularization
                RandAugment(),  # Apply RandAugment after initial geometric transformations
                self.seathru_transform,  # Apply SeaThru transformation (Ensure it doesn’t conflict with ToTensor)
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),  # Convert image to tensor (only if not done by seathru_transform)
                normalize,  # Normalize at the very end
            ])
        elif self.augmentation_strategy == "combined" and self.seathru_transform:
            print("-----------------------------COMBINED AUGMENTATION INITIALIZED-------------------------------------")
            def combined_transform(image_name, image):
                if torch.rand(1).item() <= self.combine_prob:
                    image = self.seathru_transform(image_name, image)
                return transforms.Compose([
                    RandAugment(),
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    normalize,
                ])(image)

            train_transforms = combined_transform
        else:
            print("-------------------------------------BASELINE AUGMENTATION INITIALIZED-------------------------------------")
            train_transforms = transforms.Compose([
                # RandAugment(),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomEqualize(p=0.5),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
            # Compute sample weights for oversampling
            self.sample_weights = compute_sample_weights(np.array(self.train_set.labels))

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
        # if stage == 'test' or stage is None:
        #     self.test_set = CoCoDataset(
        #         image_dir=os.path.join(self.data_dir, "eval_images"),
        #         anno_path=os.path.join(self.data_dir, "object_detection/eval.json"),
        #         input_transform=test_transforms,
        #         labels_path=os.path.join(self.data_dir, "annotations/test.npy")
        #     )

        # Update collator if using cutmix
        if self.use_cutmix:
            self.collator = CutMixCollator(self.cutmix_alpha)

        # Optional: Filter samples based on train classes
        if self.train_classes is not None:
            self.train_set.filter_samples(self.train_classes)

    def _apply_undersampling(self, dataset):
        """Reduce dataset size by under-sampling majority classes."""
        class_counts = self._get_class_counts(dataset.labels)
        min_count = min(class_counts)  # Get minimum count for any class
        sampled_indices = []

        # Group indices by class
        class_indices = {i: [] for i in range(len(class_counts))}
        for idx, label in enumerate(dataset.labels):
            for i, val in enumerate(label):
                if val:
                    class_indices[i].append(idx)

        # Sample `min_count` indices for each class
        for class_id, indices in class_indices.items():
            sampled_indices.extend(random.sample(indices, min(min_count, len(indices))))

        # Filter dataset by sampled indices
        dataset.filtered_indices = sampled_indices  # Add filtering logic 
        return dataset
    def get_num_classes(self):
        """Returns number of classes

        Returns:
            int: number of classes
        """
        return len(self.classes)

    def train_dataloader(self) -> DataLoader:
        """Creates and returns training dataloader"""
        if self.sampling_strategy == "oversample":
            # Use WeightedRandomSampler for oversampling
            sampler = WeightedRandomSampler(self.sample_weights, len(self.sample_weights))
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True,
            )
        elif self.sampling_strategy == "undersample":
            # Apply undersampling
            self.train_set = self._apply_undersampling(self.train_set)
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True,
            )
        else:
            # Default (no sampling strategy)
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True,
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
