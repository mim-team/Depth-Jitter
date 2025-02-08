import os
import random
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
# import numpy as np
from tqdm import tqdm
import torch
from torchvision.transforms.functional import to_tensor, resize
import numpy as np
np.int = int  # Alias np.int to int

category_map = {str(i): i+1 for i in range(290)}

# class CoCoDataset(data.Dataset):
#     def __init__(
#         self,
#         image_dir,
#         anno_path,
#         input_transform=None,
#         labels_path=None,
#         used_category=-1,
#         train_classes=None,
#         seathru_transform=None,  # Add this parameter
#         resize=(384, 384)  # Add resize dimensions
#     ):
#         self.coco = dset.CocoDetection(root=image_dir, annFile=anno_path)
#         self.category_map = category_map
#         self.input_transform = input_transform
#         self.labels_path = labels_path
#         self.used_category = used_category
#         self.train_classes = train_classes
#         self.seathru_transform = seathru_transform  # Initialize the transform
#         self.resize = resize  # Initialize the resize dimensions

#         self.load_labels()
#         if self.train_classes is not None:
#             self.filter_samples(self.train_classes)

#         self.resize_transform = transforms.Resize(self.resize)  # Define resize transform here
#         # print(f"Seathru Transform Initialized: {self.seathru_transform is not None}")


#     def load_labels(self):
#         if os.path.exists(self.labels_path):
#             self.labels = np.load(self.labels_path).astype(np.float64)
#             self.labels = (self.labels > 0).astype(np.float64)
#         else:
#             print("No preprocessed label file found in {}.".format(self.labels_path))
#             self.labels = []
#             for i in tqdm(range(len(self.coco))):
#                 _, annotations = self.coco[i]
#                 categories = self.getCategoryList(annotations)
#                 label = self.getLabelVector(categories)
#                 self.labels.append(label)
#             self.save_datalabels(self.labels_path)

#     def filter_samples(self, train_classes):
#         filtered_indices = []
#         for i, label in enumerate(self.labels):
#             if any(label[class_idx] == 1 for class_idx in train_classes):
#                 filtered_indices.append(i)

#         self.filtered_coco = [self.coco[i] for i in filtered_indices]
#         self.labels = [self.labels[i] for i in filtered_indices]

#     def __getitem__(self, index):
#         item = self.filtered_coco[index] if hasattr(self, 'filtered_coco') else self.coco[index]
#         image, _ = item
#         img_info = self.coco.coco.imgs[self.coco.ids[index]]
#         img_name = img_info['file_name']

#         # Convert to PIL Image before resizing
#         image = Image.fromarray(np.array(image))

#         # Resize the image to the desired dimensions
#         image = resize(image, self.resize)

#         # if self.seathru_transform:
#             # Apply the SeaThru transformation
#         # print("seathru_transform is: ", self.seathru_transform)
#         image = self.seathru_transform(img_name, np.array(image))

#         # image = to_tensor(image)

#         # Ensure the tensor is in the correct format (C, H, W)
#         if len(image.shape) == 3 and image.shape[0] != 3:
#             image = image.permute(2, 0, 1)

#         # if self.input_transform and random.random() <= 0.4:
#         #     image = self.input_transform(image)

#         return image, torch.tensor(self.labels[index])

#     def getCategoryList(self, item):
#         categories = set()
#         for t in item:
#             categories.add(t["category_id"])
#         return list(categories)

#     def getLabelVector(self, categories):
#         categories = [int(c) for c in categories]
#         label = [0] * len(self.category_map)
#         for c in categories:
#             index = self.category_map.get(str(c))
#             if index is not None and index < len(label):
#                 label[index] = 1.0
#         return label

#     def __len__(self):
#         return len(self.filtered_coco) if hasattr(self, 'filtered_coco') else len(self.coco)

#     def save_datalabels(self, outpath):
#         os.makedirs(os.path.dirname(outpath), exist_ok=True)
#         labels = np.array(self.labels)
#         np.save(outpath, labels)
class CoCoDataset(data.Dataset):
    def __init__(
        self,
        image_dir,
        anno_path,
        input_transform=None,
        labels_path=None,
        used_category=-1,
        train_classes=None,
        seathru_transform=None,  # Add this parameter
        resize=(384, 384)  # Add resize dimensions
    ):
        self.coco = dset.CocoDetection(root=image_dir, annFile=anno_path)
        self.category_map = category_map  # Ensure this is consistent with your dataset
        self.input_transform = input_transform
        self.labels_path = labels_path
        self.used_category = used_category
        self.train_classes = train_classes
        self.seathru_transform = seathru_transform  # Initialize the transform
        self.resize = resize  # Initialize the resize dimensions

        self.load_labels()
        if self.train_classes is not None:
            self.filter_samples(self.train_classes)

        self.resize_transform = transforms.Resize(self.resize)  # Define resize transform here

    def load_labels(self):
        """Load or compute multi-hot encoded labels for the dataset."""
        if os.path.exists(self.labels_path):
            # Load precomputed labels
            self.labels = np.load(self.labels_path).astype(np.float32)
        else:
            print("No preprocessed label file found in {}.".format(self.labels_path))
            self.labels = []
            for i in tqdm(range(len(self.coco))):
                _, annotations = self.coco[i]
                categories = self.getCategoryList(annotations)
                label = self.getLabelVector(categories)
                self.labels.append(label)
            self.save_datalabels(self.labels_path)

    def filter_samples(self, train_classes):
        """Filter samples to include only specified training classes."""
        filtered_indices = []
        for i, label in enumerate(self.labels):
            if any(label[class_idx] == 1 for class_idx in train_classes):
                filtered_indices.append(i)

        self.filtered_coco = [self.coco[i] for i in filtered_indices]
        self.labels = [self.labels[i] for i in filtered_indices]

    def __getitem__(self, index):
        item = self.filtered_coco[index] if hasattr(self, 'filtered_coco') else self.coco[index]
        image, _ = item
        img_info = self.coco.coco.imgs[self.coco.ids[index]]
        img_name = img_info['file_name']

        # Convert to PIL Image before resizing
        image = Image.fromarray(np.array(image))

        # Resize the image to the desired dimensions
        image = resize(image, self.resize)

        # if self.seathru_transform:
            # Apply the SeaThru transformation
        # print("seathru_transform is: ", self.seathru_transform)
        image = self.seathru_transform(img_name, np.array(image))

        # image = to_tensor(image)

        # Ensure the tensor is in the correct format (C, H, W)
        if image.ndim == 3 and image.size(0) != 3:
            image = image.permute(2, 0, 1)

        # if self.input_transform and random.random() <= 0.4:
        #     image = self.input_transform(image)

        return image, torch.tensor(self.labels[index])





    def getCategoryList(self, item):
        """Extract the category IDs from annotations."""
        categories = set()
        for t in item:
            categories.add(t["category_id"])
        return list(categories)

    def getLabelVector(self, categories):
        """
        Create a multi-hot encoded vector for the given categories.
        Args:
            categories (list): List of category IDs for a sample.
        Returns:
            list: Multi-hot encoded vector for the sample.
        """
        label = [0] * len(self.category_map)
        for c in categories:
            index = self.category_map.get(str(c))
            if index is not None and index < len(label):
                label[index] = 1.0
        return np.array(label, dtype=np.int32)  # Use np.int32 instead of np.int


    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.filtered_coco) if hasattr(self, 'filtered_coco') else len(self.coco)

    def save_datalabels(self, outpath):
        """Save multi-hot encoded labels to a file."""
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        labels = np.array(self.labels)
        np.save(outpath, labels)