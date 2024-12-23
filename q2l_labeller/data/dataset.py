# from torch.utils.data.dataloader import default_collate
# from torch.utils.data import DataLoader
# import torch
# from torchvision.transforms.functional import to_tensor, resize, pil_to_tensor
# import matplotlib.pyplot as plt
# import random
# import os
# from torch.utils.data import Dataset

# import json
# import glob
# import cv2
# import numpy as np
# from PIL import Image
# from torchvision import transforms

# def histogram_stretching(image: torch.Tensor):
#     if isinstance(image, torch.Tensor):
#         image = image.cpu().numpy()
#         # image = image.to('cuda')
    
#     image_copy = image.copy()
#     valid_mask = ~np.isnan(image_copy).any(axis=2)
    
#     stretched_image = np.zeros_like(image_copy)
    
#     for c in range(image_copy.shape[2]):
#         channel = image_copy[:, :, c]
#         valid_pixels = channel[valid_mask]
        
#         if valid_pixels.size == 0:  # Check if there are no valid pixels
#             continue  # Skip this channel if empty
        
#         # Flatten valid_pixels for percentile calculation if necessary
#         valid_pixels_flat = valid_pixels.flatten()
        
#         lower_percentile = np.percentile(valid_pixels_flat, 1)
#         upper_percentile = np.percentile(valid_pixels_flat, 99)
        
#         # Clip and stretch the valid pixel values
#         valid_pixels_clipped = np.clip(valid_pixels, lower_percentile, upper_percentile)
#         valid_pixels_stretched = (valid_pixels_clipped - lower_percentile) / (upper_percentile - lower_percentile)
        
#         # Update stretched_image with stretched valid pixels
#         stretched_channel = np.zeros_like(channel)
#         stretched_channel[valid_mask] = valid_pixels_stretched
#         stretched_image[:, :, c] = stretched_channel

#     return torch.from_numpy(stretched_image).float()


# # def compute_J(I, z, B, beta, gamma):
# #     return (I - B * (1 - torch.exp(-gamma * z))) * torch.exp(beta * z)

# # def compute_I(J,beta,z,B,gamma):
# #     return (J * torch.exp(-beta * z) + B * (1 - torch.exp(-gamma * z)))
# # def compute_I_without_J(I_orig,beta,z,z_mod,B,gamma):
# #     a = torch.exp(-beta * z)
# #     b = torch.exp(-beta * z_mod)
# #     I_mod = (I_orig * b + B * (a*(1 - torch.exp(-gamma * z_mod)) - b * (1 - torch.exp(1 - torch.exp(-gamma * z)))))/a
# #     return I_mod
# def compute_I_without_J(I_orig, beta, z, z_mod, B, gamma):
#     # Check if tensors are on the GPU, and move them if they are not
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     I_orig, beta, z, z_mod, B, gamma = [x for x in [I_orig, beta, z, z_mod, B, gamma]]

#     # Perform the calculations using PyTorch operations
#     a = torch.exp(-beta * z)
#     b = torch.exp(-beta * z_mod)
#     exp_gamma_z_mod = torch.exp(-gamma * z_mod)
#     exp_gamma_z = torch.exp(-gamma * z)


#     # Correction in the formula (removed an unnecessary 'torch.exp(1 - ...)')
#     I_mod = (I_orig * b + B * (a * (1 - exp_gamma_z_mod) - b * (1 - exp_gamma_z))) / a

#     return I_mod

# class SeaThru:
#     def __init__(self,mode='train',depth_offset=True,random_B_gamma = False,shuffle_B_gamma=True,depth_pixel=True):
#         self.mode = mode
#         self.depth_offset= depth_offset
#         self.random_B_gamma = random_B_gamma,
#         self.shuffle_B_gamma = shuffle_B_gamma,
#         self.depth_pixel = depth_pixel
    
#     def __call__(self, img_name):  # Expects a tuple of (image, depth_image)
#         # img_name, img, depth_img = img_name  # Unpack the tuple
        
#         transformed_img = seathru(img_name,self.mode,self.depth_offset,self.shuffle_B_gamma,self.depth_pixel )
#         if transformed_img is None:
#             # Handle the None case appropriately
#             raise ValueError("Transformed image is None")
            
#         return transformed_img  # Return the transformed image
    
#     def __repr__(self):
#         return self.__class__.__name__ + f'(transform_param={self.transform_param})'

# # Assuming restored_image is a PIL Image
# to_tensor_transform = transforms.ToTensor()
# def seathru(image_name,mode="train",depth_offset=True,random_B_gamma = False,shuffle_B_gamma=True,depth_pixel=True):
#     # Define the paths
#     # print(depth_offset, B_mean, gamma_mean, shuffle_B_gamma, depth_pixel)
#     # print(f"Type of image_name: {type(image_name)}")
#     depth_off = 0
#     random_B = 0
#     random_gamma = 0
#     if mode == 'train':
#         image_dir = '/home/mundus/mrahman528/thesis/fgvc-comp-2023/train'
#         depth_dir = '/home/mundus/mrahman528/thesis/Depth-Anything/depth_vis_all/'
#         json_file_path = '/home/mundus/mrahman528/thesis/sucre/parameters.json'  # Update this path
#     else:
#         image_dir = '/home/mundus/mrahman528/thesis/fgvc-comp-2023/train'
#         depth_dir = '/home/mundus/mrahman528/thesis/Depth-Anything/depth_vis_all/'
#         json_file_path = '/home/mundus/mrahman528/thesis/sucre/parameters.json'

#     # Load the JSON data containing parameters
#     with open(json_file_path, 'r') as file:
#         data = json.load(file)

#     # Initialize PyTorch device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Construct file paths
#     image_file = os.path.join(image_dir, image_name)
#     # image_file = image_name
#     depth_file = os.path.join(depth_dir, os.path.splitext(image_name)[0] + '_depth.png')
#     # Read and process the image
#     image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
#     depth_image = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = torch.tensor(image, dtype=torch.float32).to(device)
#     depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
#     # Read and process the depth image
#     # depth_image = cv2.imread(depth_image, cv2.IMREAD_UNCHANGED)
#     normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
#     distance_map = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
#     distance_map = torch.tensor(distance_map, dtype=torch.float32).to(device)
#     distance_map_gray = torch.mean(distance_map, dim=2).to(device)
#     distance_map_gray_updated = distance_map_gray
#     # Only select pixels where there is depth information
#     v_valid, u_valid = torch.where(distance_map_gray > 0)
#     image_valid = image[v_valid, u_valid].to(device)

#     #generate random depth offset
    
#     depth_off = 70

#     #generate random B and gamma
#     if random_B_gamma:
#         random_B = random.uniform(1e-6,1)
#         random_gamma = random.uniform(0.0022304835938825366, 1.4107012727967811)

#     #Change the corner of the depth map
#     if depth_pixel:
#         distance_map_gray_updated[0,0] = distance_map_gray[0,0] + random.randrange(50,255)
#         distance_map_gray_updated[-1,-1] = distance_map_gray[-1,-1] - random.randrange(50,255)
    
#     distance_map_valid = distance_map_gray[v_valid, u_valid].to(device)
#     distance_map_valid_updated = distance_map_gray_updated[v_valid, u_valid].to(device)
#     betac, gammac, Bc = [], [], []
#     if image_name in data:
#         for channel in ['channel_0', 'channel_1', 'channel_2']:
            
#             if shuffle_B_gamma:
#                 #get a random image
#                 random_image_name = random.choice(list(data.keys()))
#                 #do not change beta c
#                 betac.append(data[image_name][channel]['betac'])
#                 #change gamma and B
#                 gammac.append(data[random_image_name][channel]['gammac'])
#                 Bc.append(data[random_image_name][channel]['Bc'])
#             else:
#                 betac.append(data[image_name][channel]['betac'])
#                 gammac.append(data[image_name][channel]['gammac'])
#                 Bc.append(data[image_name][channel]['Bc'])
#     else:
#         print(f"Data for {image_name} not found in the data structure.")
#         return None
    
#     I_mod = torch.full(image.shape, torch.nan, dtype=torch.float32).to(device)

#     for channel in range(3):
#         I_mod[v_valid,u_valid,channel] = compute_I_without_J(
#             image_valid[:,channel], betac[channel] - 5e-3, distance_map_valid, distance_map_valid_updated - depth_off,Bc[channel] + random_B,gammac[channel] + random_gamma)
        
    
#     # Convert to PIL Image for return, assuming histogram_stretching is correctly implemented
#     restored_image = Image.fromarray(np.uint8(histogram_stretching(I_mod) * 255))
#     return to_tensor_transform(restored_image.resize((384,384)))
import cv2
import torch
import numpy as np
import json
import random
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, resize

def compute_I_without_J(I_orig, beta, z, z_mod, B, gamma):
    a = torch.exp(-beta * z)
    b = torch.exp(-beta * z_mod)
    exp_gamma_z_mod = torch.exp(-gamma * z_mod)
    exp_gamma_z = torch.exp(-gamma * z)
    I_mod = (I_orig * b + B * (a * (1 - exp_gamma_z_mod) - b * (1 - exp_gamma_z))) / a
    return I_mod

def histogram_stretching(image: torch.Tensor):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    image_copy = image.copy()
    valid_mask = ~np.isnan(image_copy).any(axis=2)
    
    stretched_image = np.zeros_like(image_copy)
    
    for c in range(image_copy.shape[2]):
        channel = image_copy[:, :, c]
        valid_pixels = channel[valid_mask]
        
        if valid_pixels.size == 0:
            continue
        
        valid_pixels_flat = valid_pixels.flatten()
        
        lower_percentile = np.percentile(valid_pixels_flat, 1)
        upper_percentile = np.percentile(valid_pixels_flat, 99)
        
        valid_pixels_clipped = np.clip(valid_pixels, lower_percentile, upper_percentile)
        valid_pixels_stretched = (valid_pixels_clipped - lower_percentile) / (upper_percentile - lower_percentile)
        
        stretched_channel = np.zeros_like(channel)
        stretched_channel[valid_mask] = valid_pixels_stretched
        stretched_image[:, :, c] = stretched_channel

    return torch.from_numpy(stretched_image).float()

class SeaThruAugmentation:
    def __init__(self, image_folder, depth_image_folder, depth_npy_folder, seathru_parameters_path, mode='train', resize=(384, 384)):
        self.image_folder = Path(image_folder)
        self.depth_image_folder = Path(depth_image_folder)
        self.depth_npy_folder = Path(depth_npy_folder)
        self.seathru_parameters_path = seathru_parameters_path
        self.mode = mode
        self.image_files = list(self.image_folder.glob('*.png'))
        self.resize = resize
        with open(seathru_parameters_path, 'r') as file:
            self.data = json.load(file)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to_tensor_transform = transforms.ToTensor()
        self.resize_transform = transforms.Resize(self.resize)  # Define resize transform here

    def __call__(self, image_name, image):
        depth_image_file = self.depth_image_folder / image_name
        depth_npy_file = self.depth_npy_folder / (Path(image_name).stem + '_raw_depth_meter.npy')

        if not depth_image_file.exists() or not depth_npy_file.exists():
            raise FileNotFoundError(f"Depth files not found for {image_name}. Skipping.")

        return self.compute_augmentation(image_name, image, str(depth_image_file), str(depth_npy_file))

    def compute_augmentation(self, image_name, image, depth_image_path, depth_npy_path):
        # Resize the image first
        image = resize(Image.fromarray(image), self.resize)
        image = np.array(image)

        # Load the depth image
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
        depth_image = cv2.resize(depth_image, (self.resize[1], self.resize[0]))  # Resize to match image
        depth_image = torch.tensor(depth_image, dtype=torch.float32)
        distance_map_gray_image = torch.mean(depth_image, dim=2)

        # Load the depth data from the npy file
        depth_data = np.load(depth_npy_path)
        depth_data_resized = cv2.resize(depth_data, (self.resize[1], self.resize[0]))  # Resize to match image
        distance_map_gray_npy = torch.tensor(depth_data_resized, dtype=torch.float32)

        # Ensure both depth maps are the same shape
        if distance_map_gray_image.shape != distance_map_gray_npy.shape:
            raise ValueError("Depth image and metric depth npy file must have the same shape")

        # Only select pixels where there is depth information in both sources
        v_valid, u_valid = torch.where((distance_map_gray_image > 0) & (distance_map_gray_npy > 0))
        image_valid = torch.tensor(image, dtype=torch.float32)[v_valid, u_valid]
        distance_map_valid_image = distance_map_gray_image[v_valid, u_valid]
        distance_map_valid_npy = distance_map_gray_npy[v_valid, u_valid]

        # Generate random depth offset between -20 and 20
        depth_offset = random.uniform(-4, 15)

        # Generate random B and gamma
        random_B = random.uniform(0,0.4)
        random_betac = random.uniform(0,0.04)
        # random_gamma = random.uniform(0.0022304835938825366, 1.4107012727967811)


        # Prepare betac, gammac, and Bc from the JSON data
        betac = [
            self.data[image_name]['channel_0']['betac'],
            self.data[image_name]['channel_1']['betac'],
            self.data[image_name]['channel_2']['betac']
        ]
        gammac = [
            self.data[image_name]['channel_0']['gammac'],
            self.data[image_name]['channel_1']['gammac'],
            self.data[image_name]['channel_2']['gammac']
        ]
        Bc = [
            self.data[image_name]['channel_0']['Bc'],
            self.data[image_name]['channel_1']['Bc'],
            self.data[image_name]['channel_2']['Bc']
        ]
        for i in range(len(Bc)):
            if Bc[i] == 1:
                Bc[i] -= random.uniform(0,0.4)
        # Compute the modified image using depth npy
        I_mod_npy = torch.full(image.shape, torch.nan, dtype=torch.float32)
        for channel in range(3):
            I_mod_npy[v_valid, u_valid, channel] = compute_I_without_J(
                image_valid[:, channel], betac[channel] + random_betac, distance_map_valid_npy, distance_map_valid_npy + depth_offset, Bc[channel], gammac[channel]
            )

        # Replace NaNs with 0
        I_mod_npy = torch.nan_to_num(I_mod_npy)

        # Convert to tensor and resize before returning
        restored_image_npy = I_mod_npy.cpu().numpy().clip(0, 255).astype('uint8')
        restored_image_npy = Image.fromarray(restored_image_npy)
        restored_image_npy = self.resize_transform(restored_image_npy)
        restored_image_npy = to_tensor(restored_image_npy)

        # Ensure the tensor is in the correct format (C, H, W)
        if len(restored_image_npy.shape) == 3 and restored_image_npy.shape[0] != 3:
            restored_image_npy = restored_image_npy.permute(2, 0, 1)
        return restored_image_npy
