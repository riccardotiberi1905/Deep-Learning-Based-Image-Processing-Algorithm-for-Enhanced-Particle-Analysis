import os
import h5py
import numpy as np
import cv2
from glob import glob
import random
from patchify import patchify
from skimage.measure import label, regionprops


'''
This script is designed to preprocess a set of training images and corresponding masks and then save the resulting patches into two 
HDF5 files—one for the processed image patches (inputs) and one for the corresponding mask patches (targets). 
Here’s what happens step by step:

Data Loading and Preprocessing:

The script searches for all JPEG images in two folders (one for images and one for masks).
For each image, it extracts the green channel and resizes it to half the original dimensions.
The mask images are also resized to the same dimensions.
Patch Extraction:

Two methods are used to generate training samples:
Keypoint-based sampling (expand_true_false_sample):
It uses an ORB detector on the mask to find keypoints.
For each keypoint, it extracts a 64×64 patch (using a radius r=32) from both the processed green channel and the mask.
It applies a random offset (up to ±50 pixels) for each patch, with boundary checks to ensure valid crops.
Random sampling (get_random_sample):
It randomly selects coordinates (within safe margins) and extracts a 64×64 patch from both the image and the mask.

Dataset Formation:
The patches from both sampling methods are concatenated into two NumPy arrays: one for the input data and one for the target masks.
The arrays are reshaped by adding an extra dimension (likely to serve as a channel dimension) so that each patch has shape similar to (1, 64, 64).

Saving the Data:
The script then creates two HDF5 files—one named train_input1.h5 (for the image patches) and one named train_target1.h5 (for the mask patches).
Each patch is stored as an individual dataset within these HDF5 files, with the dataset names being string versions of their index (e.g., "0", "1", …).

Console Outputs:
Throughout the process, the script prints the number of keypoints detected (per image) and an incremental index for each valid patch extracted.
Finally, it prints the shape of the concatenated datasets for both inputs and targets.

Expected Outcome:
Two HDF5 files are generated containing your training data: one for the processed green channel patches and one for the corresponding mask patches.
The printed shapes will indicate the total number of patches (e.g., something like (n, 1, 64, 64) for both the input and target arrays), where n is the sum of patches from both sampling methods.
These files are ready to be used for training a neural network (such as the MSR-NET mentioned in your paths).
'''

#------------------------ Essential Functions ------------------------------
def true_false_sample_exp(random_count=1):
    train_image_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\1_class_tloss_FOV_small\train\images"
    train_mask_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\1_class_tloss_FOV_small\train\masks"

    file_path_im = sorted(glob(os.path.join(train_image_path, "*.png")))

    train_data = []
    label_data = []
    r = 32
    index = 0
    for im_path in file_path_im:

        name = os.path.basename(im_path)
        image_file = os.path.join(train_image_path, name)
        mask_file = os.path.join(train_mask_path, name)

        # Read image and mask
        im = cv2.imread(image_file)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        h, w, _ = im.shape

        h_m, w_m = mask.shape

        if not h == h_m and w == w_m:
            raise ValueError(f'Shape fail in {name}')

        # Compute green channel of image
        green_channels = im[:, :, 1]

        # Eliminate unwanted pixel intensities
        mask = np.where(mask >= 128, 255, 0).astype(np.uint8)

        # Label connected components
        label_mask = label(mask, connectivity=1)
        particles_mask = regionprops(label_mask)

        for particle in particles_mask:
            y, x = particle.centroid  # Extract centroid coordinates

            for _ in range(random_count):
                inde_x = random.randint(-20, 20)
                inde_y = random.randint(-20, 20)
                h_head, h_end = int(y) - r + inde_x, int(y) + r + inde_x
                w_head, w_end = int(x) - r + inde_y, int(x) + r + inde_y
                flage = True

                # Ensure patch stays within bounds
                if h_head < 0 or h_end > h or w_head < 0 or w_end > w:
                    h_head, h_end = int(y) - r, int(y) + r
                    w_head, w_end = int(x) - r, int(x) + r

                    if h_head < 0 or h_end > h or w_head < 0 or w_end > w:
                        flage = False
                        pass
                    else:
                        crop_mask = mask[h_head:h_end, w_head:w_end]
                        g_crop_img = green_channels[h_head:h_end, w_head:w_end]
                else:
                    crop_mask = mask[h_head:h_end, w_head:w_end]
                    g_crop_img = green_channels[h_head:h_end, w_head:w_end]

                if flage:
                    # cv2.imshow("P", crop_mask)
                    # cv2.imshow("T", g_crop_img)
                    # cv2.waitKey(0)
                    train_data.append(g_crop_img)
                    label_data.append(crop_mask)
                    index += 1
                    #print(f"Patch {index} extracted.")

    train_data = np.array(train_data)
    train_data = np.expand_dims(train_data, 1)
    label_data = np.array(label_data)
    label_data = np.expand_dims(label_data, 1)
    print(train_data.shape, label_data.shape)
    return train_data, label_data

def true_false_sample_exp_2_classes(random_count=1, random_count_dirt=1):
    train_image_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\2_classes_ce_dice_FOV_small\train\images"
    train_mask_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\2_classes_ce_dice_FOV_small\train\masks"

    file_path_im = sorted(glob(os.path.join(train_image_path, "*.png")))

    train_data = []
    label_data = []
    r = 32
    index = 0
    for im_path in file_path_im:

        name = os.path.basename(im_path)
        image_file = os.path.join(train_image_path, name)
        mask_file = os.path.join(train_mask_path, name)

        # Read image and mask
        im = cv2.imread(image_file)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        h, w, _ = im.shape

        h_m, w_m = mask.shape

        if not h == h_m and w == w_m:
            raise ValueError(f'Shape fail in {name}')

        # Compute green channel of image
        green_channels = im[:, :, 1]

        middle_class_range = (80, 100)  # Pixels in this range will be set to 90
        mask_bin = np.zeros_like(mask)  # Start with all zeros
        mask_bin[(mask >= middle_class_range[0]) & (mask <= middle_class_range[1])] = 90  # Middle class to 90
        mask_bin[mask > 128] = 255  # Above 128 to 255

        # Label connected components
        label_mask = label(mask_bin == 255, connectivity=1)
        particles_mask = regionprops(label_mask)

        for particle in particles_mask:
            y, x = particle.centroid  # Extract centroid coordinates

            for _ in range(random_count):
                inde_x = random.randint(-20, 20)
                inde_y = random.randint(-20, 20)
                h_head, h_end = int(y) - r + inde_x, int(y) + r + inde_x
                w_head, w_end = int(x) - r + inde_y, int(x) + r + inde_y
                flage = True

                # Ensure patch stays within bounds
                if h_head < 0 or h_end > h or w_head < 0 or w_end > w:
                    h_head, h_end = int(y) - r, int(y) + r
                    w_head, w_end = int(x) - r, int(x) + r

                    if h_head < 0 or h_end > h or w_head < 0 or w_end > w:
                        flage = False
                        pass
                    else:
                        crop_mask = mask[h_head:h_end, w_head:w_end]
                        g_crop_img = green_channels[h_head:h_end, w_head:w_end]
                else:
                    crop_mask = mask[h_head:h_end, w_head:w_end]
                    g_crop_img = green_channels[h_head:h_end, w_head:w_end]

                if flage:
                    # cv2.imshow("P", crop_mask)
                    # cv2.imshow("T", g_crop_img)
                    # cv2.waitKey(0)
                    train_data.append(g_crop_img)
                    label_data.append(crop_mask)
                    index += 1
                    #print(f"Patch {index} extracted.")

        # Label connected components for dirt class (value 90)
        label_mask = label(mask_bin == 90, connectivity=1)
        dirt_regions = regionprops(label_mask)

        points_per_region = 20  # Number of random points per dirt region

        for region in dirt_regions:
            coords = region.coords  # Get all pixel positions in the dirt region

            if len(coords) < points_per_region:
                selected_points = coords  # If region is small, take all points
            else:
                selected_points = coords[np.random.choice(len(coords), points_per_region, replace=False)]

            for (y, x) in selected_points:
                for _ in range(random_count_dirt):
                    inde_x = random.randint(-20, 20)
                    inde_y = random.randint(-20, 20)
                    h_head, h_end = int(y) - r + inde_x, int(y) + r + inde_x
                    w_head, w_end = int(x) - r + inde_y, int(x) + r + inde_y
                    flag = True

                    # Ensure patch stays within bounds
                    if h_head < 0 or h_end > h or w_head < 0 or w_end > w:
                        h_head, h_end = int(y) - r, int(y) + r
                        w_head, w_end = int(x) - r, int(x) + r

                        if h_head < 0 or h_end > h or w_head < 0 or w_end > w:
                            flag = False
                            continue  # Skip if out of bounds

                    crop_mask = mask[h_head:h_end, w_head:w_end]
                    g_crop_img = green_channels[h_head:h_end, w_head:w_end]

                    if flag:
                        train_data.append(g_crop_img)
                        label_data.append(crop_mask)
                        index += 1
                        # print(f"Patch {index} extracted.")

    train_data = np.array(train_data)
    train_data = np.expand_dims(train_data, 1)
    label_data = np.array(label_data)
    label_data = np.expand_dims(label_data, 1)
    print(train_data.shape, label_data.shape)
    return train_data, label_data

def patch_generator_outdated():
    # Define paths for images and masks
    train_image_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\train\images"
    train_mask_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\train\masks"

    file_path_im = sorted(glob(os.path.join(train_image_path, "*.jpg")))
    file_path_mask = sorted(glob(os.path.join(train_mask_path, "*.jpg")))

    # Check if both lists have the same length
    if len(file_path_im) != len(file_path_mask):
        raise ValueError("The number of images and masks do not match!")

    train_data = []
    label_data = []

    # Parameters for patch extraction using patchify:
    # Patch size of 64x64, non-overlapping (step equal to patch size)
    patch_size = (64, 64)
    step = 64

    for im_path in file_path_im:

        name = os.path.basename(im_path)
        image_file = os.path.join(train_image_path, name)
        mask_file = os.path.join(train_mask_path, name)

        # Load image and mask
        im = cv2.imread(image_file)  # Loaded in BGR format
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale

        h, w, _ = im.shape

        h_m, w_m = mask.shape

        if not h == h_m and w == w_m:
            raise ValueError(f'Shape fail in {name}')

        # Extract the green channel from the image (assuming BGR, green is index 1)
        green_channel = im[:, :, 1]

        # Threshold mask: convert to binary (0 or 255)
        mask = np.where(mask >= 128, 255, 0).astype(np.uint8)

        # Generate patches for image and mask using patchify
        patches_green = patchify(green_channel, patch_size, step=step)
        # patches_mask = patchify(mask, patch_size, step=step)
        patches_mask = patchify(mask, patch_size, step=step)

        # Flatten the patches arrays
        flat_green = patches_green.reshape(-1, 64, 64)
        flat_mask = patches_mask.reshape(-1, 64, 64)

        # Append corresponding patches to the lists
        for patch_green, patch_mask in zip(flat_green, flat_mask):
            train_data.append(patch_green)
            label_data.append(patch_mask)

    # Convert lists to NumPy arrays and add a channel dimension
    train_data = np.array(train_data)
    train_data = np.expand_dims(train_data, axis=1)
    label_data = np.array(label_data)
    label_data = np.expand_dims(label_data, axis=1)

    print("Training patches shape:", train_data.shape)
    print("Label patches shape:", label_data.shape)
    return train_data, label_data

def patch_generator_updated(test_img_data, test_mask_data):
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

def paint_border(files_img, files_masks, test_imgs_path, test_masks_path, img_save_path, mask_save_path, desired_h=1088, desired_w=1088):
    #desired_h ; desired_w: Height and width divisible by 64x64 patches
    train_img_data = []
    train_mask_data = []

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
        new_mask = np.select([new_mask == 0], [90], new_mask)

        # Compute offsets to center the image in the padded array
        top = (desired_h - h) // 2
        left = (desired_w - w) // 2

        # Copy original image and mask into the centered position of the padded arrays
        new_img[top:top + h, left:left + w] = green_channel
        new_mask[top:top + h, left:left + w] = mask_bin

        output_path_img = os.path.join(img_save_path, name)
        cv2.imwrite(output_path_img, new_img.astype(np.uint8))
        output_path_mask = os.path.join(mask_save_path, name)
        cv2.imwrite(output_path_mask, new_mask.astype(np.uint8))

        train_img_data.append(new_img)
        train_mask_data.append(new_mask)

        # Convert lists to NumPy arrays and add a channel dimension
    train_img_data = np.array(train_img_data)
    train_img_data = np.expand_dims(train_img_data, axis=1)  # shape becomes (N, 1, H, W)
    train_mask_data = np.array(train_mask_data)
    train_mask_data = np.expand_dims(train_mask_data, axis=1)

    print("Test padded images shape:", train_img_data.shape)
    print("Test padded masks shape:", train_mask_data.shape)

    return train_img_data, train_mask_data

#------------------------------ Define Directories ----------------------------
train_images_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\1_class_tloss_FOV_small\train\images"
train_masks_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\1_class_tloss_FOV_small\train\masks"

img_save_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\1_class_tloss_FOV_small\train_padded_imgs"
mask_save_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\1_class_tloss_FOV_small\train_padded_masks"

#Define Base Directory to store H5 files
base_dir = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\1_class_tloss_FOV_small"

#------------------------------ Run Script -------------------------------------
#x1, y1 = expand_true_false_sample(random_count=1)
x1, y1 = true_false_sample_exp(random_count=20)

train_images = sorted(glob(os.path.join(train_images_path, "*.png")))  # Change extension if needed
train_masks = sorted(glob(os.path.join(train_masks_path, "*.png")))  # Change extension if needed

train_img_data, train_mask_data = paint_border(train_images, train_masks,
                                             train_images_path, train_masks_path,
                                             img_save_path, mask_save_path,
                                             desired_h=896, desired_w=896)

#x2, y2 = get_random_sample(random_count=10) #Try with 200, 500, 600, 1000... Try to have balanced data
x2, y2 = patch_generator_updated(train_img_data, train_mask_data)

dataset = np.concatenate((x1, x2), axis=0)
target = np.concatenate((y1, y2), axis=0)
print(dataset.shape)
print(target.shape)

#Save to h5 directory
save_target_path = os.path.join(base_dir, 'train_target.h5')
save_input_path = os.path.join(base_dir, 'train_input.h5')

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