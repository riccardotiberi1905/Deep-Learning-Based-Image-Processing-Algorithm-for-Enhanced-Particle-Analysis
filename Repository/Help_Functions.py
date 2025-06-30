import h5py
import numpy as np
from patchify import unpatchify

def compute_iou(prediction, targets, num_classes=2):
    """
    Compute IoU for each class and return as a dict.
    prediction: torch.Tensor, shape (batch, H, W) (predicted class indices)
    targets: torch.Tensor, shape (batch, H, W) (ground truth class indices)
    """
    iou_scores = {}
    for cls in range(num_classes):
        pred_mask = (prediction == cls)
        target_mask = (targets == cls)
        intersection = np.sum(pred_mask & target_mask)
        union = np.sum(pred_mask | target_mask)

        iou_scores[cls] = intersection / (union + 1e-8)
    return iou_scores

def compute_dice(prediction, targets, num_classes=2):
    """
    Compute Dice coefficient for each class and return as a dict.
    prediction: torch.Tensor, shape (batch, H, W) (predicted class indices)
    targets: torch.Tensor, shape (batch, H, W) (ground truth class indices)
    """
    dice_scores = {}
    for cls in range(num_classes):
        pred_mask = (prediction == cls)
        target_mask = (targets == cls)
        intersection = np.sum(pred_mask & target_mask)
        dice = (2 * intersection) / (np.sum(pred_mask) + np.sum(target_mask) + 1e-8)
        dice_scores[cls] = dice
    return dice_scores

def extract_ordered_h5_testing(images_path, mask_path):
    """
    Extract order image and mask patches from h5 files for further reconstruction
    """
    with h5py.File(images_path, 'r') as f:
        keys = sorted(list(f.keys()), key=lambda x: int(x))
        # For each key, load the dataset (assumed shape: (channels, height, width))
        samples = [f[key][...] for key in keys]
        # Stack samples along a new axis (axis=0 gives shape: (batch, channels, height, width))
        image_patches = np.stack(samples, axis=0)
        print("Input Images Shape:", image_patches.shape)

    with h5py.File(mask_path, 'r') as f:
        keys = sorted(list(f.keys()), key=lambda x: int(x))
        # For each key, load the dataset (assumed shape: (channels, height, width))
        samples = [f[key][...] for key in keys]
        # Stack samples along a new axis (axis=0 gives shape: (batch, channels, height, width))
        mask_patches = np.stack(samples, axis=0)
        print("Target Masks Shape:", mask_patches.shape)

    return image_patches, mask_patches

def extract_ordered_h5_images(images_path):
    """
    Extract order image and mask patches from h5 files for further reconstruction
    """
    with h5py.File(images_path, 'r') as f:
        keys = sorted(list(f.keys()), key=lambda x: int(x))
        # For each key, load the dataset (assumed shape: (channels, height, width))
        samples = [f[key][...] for key in keys]
        # Stack samples along a new axis (axis=0 gives shape: (batch, channels, height, width))
        image_patches = np.stack(samples, axis=0)
        print("Input Images Shape:", image_patches.shape)

    return image_patches

def recompone(test_img_patches, test_mask_patches, prediction_patches, full_img_shape=(1088, 1088), patch_size=(64, 64)):
    """
    Reconstruct full images and masks from non-overlapping patches.

    Args:
        test_img_patches (np.ndarray): Array of image patches with shape
            (total_patches, 1, patch_h, patch_w).
        test_mask_patches (np.ndarray): Array of mask patches with shape
            (total_patches, 1, patch_h, patch_w).
        prediction_patches (np.ndarray): Array of prediction outputs patches with shape
            (total_patches, 1, patch_h, patch_w).
        full_img_shape (tuple): Shape (height, width) of the padded full images.
        patch_size (tuple): Size (patch_h, patch_w) of each patch.

    Returns:
        reconstructed_imgs (np.ndarray): Reconstructed images with shape (N, 1, H, W).
        reconstructed_masks (np.ndarray): Reconstructed masks with shape (N, 1, H, W).
        reconstructed_predictions (np.ndarray): Reconstructed prediction outputs with shape (N, 1, H, W).
    """
    # Compute grid dimensions.
    full_h, full_w = full_img_shape
    patch_h, patch_w = patch_size
    grid_h = full_h // patch_h  # e.g., 1088/64 = 17
    grid_w = full_w // patch_w  # e.g., 17
    patches_per_img = grid_h * grid_w

    # Determine number of full images.
    total_patches = test_img_patches.shape[0]
    if total_patches % patches_per_img != 0:
        raise ValueError("Total number of patches is not divisible by patches per image.")
    N_full_imgs = total_patches // patches_per_img

    reconstructed_imgs = []
    reconstructed_masks = []
    reconstructed_predictions = []

    for i in range(N_full_imgs):
        # Get the patches for the i-th image.
        start = i * patches_per_img
        end = (i + 1) * patches_per_img

        # Each patch is assumed to be (1, 64, 64) - squeeze to get (64,64)
        img_patches = np.squeeze(test_img_patches[start:end], axis=1)  # shape: (patches_per_img, 64, 64)
        mask_patches = np.squeeze(test_mask_patches[start:end], axis=1)  # shape: (patches_per_img, 64, 64)
        #pred_patches = np.squeeze(prediction_patches[start:end], axis=1)  # shape: (patches_per_img, 64, 64)
        pred_patches = prediction_patches[start:end]  # shape: (patches_per_img, 64, 64)

        # Reshape the flat list of patches into a grid.
        img_grid = img_patches.reshape(grid_h, grid_w, patch_h, patch_w)
        mask_grid = mask_patches.reshape(grid_h, grid_w, patch_h, patch_w)
        prediction_grid = pred_patches.reshape(grid_h, grid_w, patch_h, patch_w)

        # Reconstruct the full image from the grid.
        recon_img = unpatchify(img_grid, full_img_shape)
        recon_mask = unpatchify(mask_grid, full_img_shape)
        recon_prediction = unpatchify(prediction_grid, full_img_shape)

        # Add a channel dimension back (to have shape (1, H, W))
        recon_img = np.expand_dims(recon_img, axis=0)
        recon_mask = np.expand_dims(recon_mask, axis=0)
        recon_prediction = np.expand_dims(recon_prediction, axis=0)

        reconstructed_imgs.append(recon_img)
        reconstructed_masks.append(recon_mask)
        reconstructed_predictions.append(recon_prediction)

    # Convert lists to numpy arrays. Final shapes: (N, 1, H, W)
    reconstructed_imgs = np.array(reconstructed_imgs)
    reconstructed_masks = np.array(reconstructed_masks)
    reconstructed_predictions = np.array(reconstructed_predictions)

    print("Reconstructed images shape:", reconstructed_imgs.shape)
    print("Reconstructed masks shape:", reconstructed_masks.shape)
    print("Reconstructed predictions shape:", reconstructed_predictions.shape)

    return reconstructed_imgs, reconstructed_masks, reconstructed_predictions

def recompone_prediction(prediction_patches, full_img_shape=(1088, 1088), patch_size=(64, 64)):
    """
    Reconstruct full images and masks from non-overlapping patches.

    Args:
        prediction_patches (np.ndarray): Array of prediction outputs patches with shape
            (total_patches, 1, patch_h, patch_w).
        full_img_shape (tuple): Shape (height, width) of the padded full images.
        patch_size (tuple): Size (patch_h, patch_w) of each patch.

    Returns:
        reconstructed_predictions (np.ndarray): Reconstructed prediction outputs with shape (N, 1, H, W).
    """
    # Compute grid dimensions.
    full_h, full_w = full_img_shape
    patch_h, patch_w = patch_size
    grid_h = full_h // patch_h  # e.g., 1088/64 = 17
    grid_w = full_w // patch_w  # e.g., 17
    patches_per_img = grid_h * grid_w

    # Determine number of full images.
    total_patches = prediction_patches.shape[0]
    if total_patches % patches_per_img != 0:
        raise ValueError("Total number of patches is not divisible by patches per image.")
    N_full_imgs = total_patches // patches_per_img

    reconstructed_predictions = []

    for i in range(N_full_imgs):
        # Get the patches for the i-th image.
        start = i * patches_per_img
        end = (i + 1) * patches_per_img

        # Each patch is assumed to be (1, 64, 64) - squeeze to get (64,64)
        pred_patches = prediction_patches[start:end]  # shape: (patches_per_img, 64, 64)

        # Reshape the flat list of patches into a grid.
        prediction_grid = pred_patches.reshape(grid_h, grid_w, patch_h, patch_w)

        # Reconstruct the full image from the grid.
        recon_prediction = unpatchify(prediction_grid, full_img_shape)

        # Add a channel dimension back (to have shape (1, H, W))
        recon_prediction = np.expand_dims(recon_prediction, axis=0)

        reconstructed_predictions.append(recon_prediction)

    # Convert lists to numpy arrays. Final shapes: (N, 1, H, W)
    reconstructed_predictions = np.array(reconstructed_predictions)

    print("Reconstructed predictions shape:", reconstructed_predictions.shape)

    return reconstructed_predictions