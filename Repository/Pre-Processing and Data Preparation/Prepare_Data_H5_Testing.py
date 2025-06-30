import h5py
import cv2
import os
from glob import glob
import numpy as np
from patchify import patchify

'''
Debugging script --> "H5_Dataset_Testing"
'''

'''
This script prepares test images and their corresponding segmentation ground truth for evaluation. 

Here’s what happens step by step:

1.	Loads test images.
2.	Loads corresponding ground truth masks (testing).
3.	Pads zeros to images to all borders to make them 1088x1088. Same for masks.
4.  Divides images in 64x64 patches (with no overlapping). 
4.	Saves test data in HDF5 format (.h5) for easy loading.

--> Provides a recompone function to later reconstruct the image after real testing.
'''

test_img_leyre_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\2_classes_ce_dice\test\images"
test_mask_leyre_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\2_classes_ce_dice\test\masks"

test_img_leyre_files = sorted(glob(os.path.join(test_img_leyre_path, "*.png")))  # Assuming masks are JPG
test_mask_leyre_files = sorted(glob(os.path.join(test_mask_leyre_path, "*.png")))  # Assuming masks are JPG


def paint_border(files_img, files_masks, test_imgs_path, test_masks_path, desired_h=1088, desired_w=1088):
    #desired_h ; desired_w: Height and width divisible by 64x64 patches
    test_img_data = []
    test_mask_data = []

    # Check if both lists have the same length
    if len(files_img) != len(files_masks):
        raise ValueError("The number of images and masks do not match!")

    for image_file in files_img:
        name = os.path.basename(image_file)
        # Build full paths
        full_img_path = os.path.join(test_imgs_path, name)
        full_mask_path = os.path.join(test_masks_path, name)

        # Load image and mask
        im = cv2.imread(full_img_path)  # BGR format
        mask = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)  # Grayscale

        if im is None or mask is None:
            raise ValueError(f"Error reading {name}")

        # Extract the green channel from the image (BGR, so index 1)
        green_channel = im[:, :, 1]

        # Threshold mask: convert to binary (0 or 255)
        # mask_bin = np.where(mask >= 128, 255, 0).astype(np.uint8)

        # Define thresholds
        middle_class_range = (80, 100)  # Pixels in this range will be set to 90

        # Threshold mask: convert to binary (0 or 255)
        mask_bin = np.zeros_like(mask)  # Start with all zeros
        mask_bin[(mask >= middle_class_range[0]) & (mask <= middle_class_range[1])] = 90  # Middle class to 90
        mask_bin[mask > 128] = 255  # Above 128 to 255

        # Get original dimensions
        h, w = green_channel.shape
        h_m, w_m = mask_bin.shape

        if h != h_m or w != w_m:
            raise ValueError(f'Shape mismatch in {name}: image {green_channel.shape} vs mask {mask_bin.shape}')

        # Create padded arrays filled with zeros (black)
        new_img = np.zeros((desired_h, desired_w), dtype=green_channel.dtype)
        new_img = np.select([new_img == 0], [255], new_img)
        new_mask = np.zeros((desired_h, desired_w), dtype=mask_bin.dtype)
        new_mask = np.select([new_mask == 0], [90], new_mask) # Comment if needed

        # Compute offsets to center the image in the padded array
        top = (desired_h - h) // 2
        left = (desired_w - w) // 2

        # Copy original image and mask into the centered position of the padded arrays
        new_img[top:top + h, left:left + w] = green_channel
        new_mask[top:top + h, left:left + w] = mask_bin

        test_img_data.append(new_img)
        test_mask_data.append(new_mask)

    # Convert lists to NumPy arrays and add a channel dimension
    test_img_data = np.array(test_img_data)
    test_img_data = np.expand_dims(test_img_data, axis=1)  # shape becomes (N, 1, H, W)
    test_mask_data = np.array(test_mask_data)
    test_mask_data = np.expand_dims(test_mask_data, axis=1)

    print("Test padded images shape:", test_img_data.shape)
    print("Test padded masks shape:", test_mask_data.shape)

    return test_img_data, test_mask_data


def patch_generator(test_img_data, test_mask_data):
    # Check that both arrays are 4D and have the same shape
    assert test_img_data.ndim == 4, "Test image data must be a 4D array"
    assert test_mask_data.ndim == 4, "Test mask data must be a 4D array"
    assert test_img_data.shape == test_mask_data.shape, "Image and mask shapes must match"

    # Parameters for patch extraction using patchify:
    # Patch size of 64x64, non-overlapping (step equal to patch size)
    patch_size = (64, 64)
    step = 64

    test_img_patches = []
    test_mask_patches = []

    # Iterate over each padded image/mask pair
    N = test_img_data.shape[0]
    for i in range(N):
        # Extract the 2D image (we assume the channel is in axis 1)
        image = test_img_data[i, 0, :, :]  # shape: (H, W)
        mask = test_mask_data[i, 0, :, :]  # shape: (H, W)

        # Generate patches using patchify (patchify returns a higher-dimensional array)
        patches_img = patchify(image, patch_size, step=step)
        patches_mask = patchify(mask, patch_size, step=step)

        # Flatten the patches arrays to get a list of patches per image
        flat_img = patches_img.reshape(-1, patch_size[0], patch_size[1])
        flat_mask = patches_mask.reshape(-1, patch_size[0], patch_size[1])

        # Append corresponding patches to the lists
        for patch_img, patch_mask in zip(flat_img, flat_mask):
            test_img_patches.append(patch_img)
            test_mask_patches.append(patch_mask)

    # Convert lists to NumPy arrays and add a channel dimension so final shape is (total_patches, 1, 64, 64)
    test_img_patches = np.array(test_img_patches)
    test_img_patches = np.expand_dims(test_img_patches, axis=1)
    test_mask_patches = np.array(test_mask_patches)
    test_mask_patches = np.expand_dims(test_mask_patches, axis=1)

    print("Test image patches shape:", test_img_patches.shape)
    print("Test mask patches shape:", test_mask_patches.shape)
    return test_img_patches, test_mask_patches


test_img_data, test_mask_data = paint_border(test_img_leyre_files, test_mask_leyre_files,
                                             test_img_leyre_path, test_mask_leyre_path, desired_h=1088, desired_w=1088)

test_img_patches, test_mask_patches = patch_generator(test_img_data, test_mask_data)

dataset = test_img_patches
target = test_mask_patches
print(dataset.shape)
print(target.shape)

#Define Base Directory to store H5 files
base_dir = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\2_classes_ce_dice"
save_target_path = os.path.join(base_dir, 'test_target.h5')
save_input_path = os.path.join(base_dir, 'test_input.h5')

target_h5f = h5py.File(save_target_path, 'w')
input_h5f = h5py.File(save_input_path, 'w')
total_number = len(dataset)
for i in range(total_number):

    #print(i)
    target_data = target[i, ...]
    target_h5f.create_dataset(str(i), data=target_data)

    input_data = dataset[i, ...]
    input_h5f.create_dataset(str(i), data=input_data)
target_h5f.close()
input_h5f.close()