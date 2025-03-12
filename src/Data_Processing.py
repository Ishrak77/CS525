import os
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from torch.utils.data import Dataset


class UNetDataset(Dataset):
    
    def __init__(self, train_files, data_dir, image_dim=(128, 128)):
        """
        Dataset for loading images and masks from HDF5 files.

        Parameters:
        - train_files: List of file names of the training HDF5 files.
        - data_dir: Directory where the HDF5 files are stored.
        - image_dim: Desired dimensions of the output images and masks (height, width).
        """
        self.train_files = train_files
        self.data_dir = data_dir
        self.image_dim = image_dim
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.train_files)
    
    def __getitem__(self, idx):
        """
        Load a single sample from the HDF5 file, apply transformations, and return the image and mask.
        
        Parameters:
        - idx: Index of the sample to be loaded.
        
        Returns:
        - image: Tensor of shape (C, H, W) for the image.
        - mask: Tensor of shape (1, H, W) for the mask.
        """
        file_path = os.path.join(self.data_dir, self.train_files[idx])
        with h5py.File(file_path, 'r') as hf:
            # Check for the correct keys
            if 'image' in hf.keys() and 'mask' in hf.keys():
                image = hf['image'][:]  # Expected shape: (H, W, 4)
                mask = hf['mask'][:]    # Expected shape: (H, W) or (H, W, C)
                # Select only the first two planes for the image
                image = image[..., :2]  # Shape: (H, W, 2)
    
                # Select only the second plane for the mask
                mask = mask[..., 1:2]  # Shape: (H, W, 1) to keep dims consistent
            else:
                raise KeyError(f"Unexpected keys in {self.train_files[idx]}: {list(hf.keys())}")

            # Resize images if necessary
            if image.shape[:2] != self.image_dim:
                image = resize(
                    image, 
                    (*self.image_dim, image.shape[2]), 
                    preserve_range=True, 
                    anti_aliasing=True
                )

            if mask.shape != self.image_dim:
                # Use nearest-neighbor interpolation for masks to preserve labels
                mask = resize(
                    mask, 
                    self.image_dim, 
                    preserve_range=True, 
                    order=0, 
                    anti_aliasing=False
                )

            # Normalize the image
            image_max = np.max(image)
            if image_max > 0:
                image = image.astype('float32') / image_max
            else:
                image = image.astype('float32')

            # Normalize the mask
            mask_max = np.max(mask)
            if mask_max > 0:
                mask = mask.astype('float32') / mask_max
            else:
                mask = mask.astype('float32')  # All zeros

            # Convert to tensor and reorder channels to (C, H, W)
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)  # C, H, W
                
            mask = torch.tensor(mask.transpose(2, 0, 1), dtype=torch.float32)  # C, H, W

        return image, mask


def show_single_h5_file(filename):
    # Load the image and mask from the HDF5 file
    with h5py.File(filename, 'r') as h5_file:
        image = h5_file["image"][:]  # Shape: (H, W, 4)
        mask = h5_file["mask"][:]    # Shape: (H, W, 3)

    # --- Figure 1: T1, T1Gd, T1 + NCR, and T1Gd + ET ---
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    overlay_pairs = [
        (image[..., 0], None, "T1", None),       # T1
        (image[..., 1], None, "T1Gd", None),     # T1Gd
        (image[..., 0], mask[..., 0], "T1 + NCR", "Greens"),  # T1 + NCR
        (image[..., 1], mask[..., 2], "T1Gd + ET", "Reds")   # T1Gd + ET
    ]

    for i, (img, msk, title, cmap) in enumerate(overlay_pairs):
        axs[i].imshow(img, cmap="gray")  # Show base MRI image
        if msk is not None:
            axs[i].imshow(msk, cmap=cmap, alpha=0.3)  # Overlay mask
        axs[i].set_title(title)
        axs[i].axis('off')

    fig.suptitle('T1, T1Gd, T1 + NCR, and T1Gd + ET')
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    overlay_pairs = [
        (image[..., 2], None, "T2-Weighted", None),      # T2-Weighted
        (image[..., 3], None, "FLAIR", None),            # FLAIR
        (image[..., 2], mask[..., 1], "T2-Weighted + ED", "Greens"),  # T2-Weighted + ED
        (image[..., 3], mask[..., 1], "FLAIR + ED", "Greens")         # FLAIR + ED
    ]

    for i, (img, msk, title, cmap) in enumerate(overlay_pairs):
        axs[i].imshow(img, cmap="gray")  # Show base MRI image
        if msk is not None:
            axs[i].imshow(msk, cmap=cmap, alpha=0.3)  # Overlay mask
        axs[i].set_title(title)
        axs[i].axis('off')

    fig.suptitle('T2-Weighted, FLAIR, T2-Weighted + ED, and FLAIR + ED')
    plt.tight_layout()
    plt.show()

def get_all_slices(volume_index, slice_range, data_dir):
    """
    Generates filenames for the MRI volume and slice range.

    Parameters:
    - volume_index: Index of the volume (e.g., 1 for 'volume_1', 2 for 'volume_2', etc.)
    - slice_range: List of slice indices to load (e.g., [0, 1, 2, ..., 155])
    - data_dir: Directory containing the HDF5 files.

    Returns:
    - filenames: List of filenames corresponding to the specified volume and slice range.
    """
    image_list = []
    mask_list=[]
    for slice_idx in slice_range:
        # Construct the filename
        filename = os.path.join(data_dir, f"volume_{volume_index}_slice_{slice_idx}.h5")
        
        # Check if the file exists before adding to the list
        if os.path.exists(filename):
            with h5py.File(filename, 'r') as h5_file:
                image_list.append(h5_file["image"][:])
                mask_list.append(h5_file["mask"][:])
        else:
            print(f"File {filename} does not exist.")
    
    return np.transpose(np.array(image_list), (1, 2, 0, 3)), np.transpose(np.array(mask_list), (1, 2, 0, 3))