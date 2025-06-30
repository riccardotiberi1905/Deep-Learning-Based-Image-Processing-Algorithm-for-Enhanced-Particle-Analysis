import h5py
import numpy as np
from skimage.transform import hough_circle, hough_circle_peaks
import cv2
import os
from glob import glob
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

#------------------------ Essential Functions ---------------------------------

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
        print("Prediction Images Shape:", image_patches.shape)

    with h5py.File(mask_path, 'r') as f:
        keys = sorted(list(f.keys()), key=lambda x: int(x))
        # For each key, load the dataset (assumed shape: (channels, height, width))
        samples = [f[key][...] for key in keys]
        # Stack samples along a new axis (axis=0 gives shape: (batch, channels, height, width))
        mask_patches = np.stack(samples, axis=0)
        print("Target Masks Shape:", mask_patches.shape)

    return image_patches, mask_patches

def detect_centered_circle(img):
    """
    Detect a circle in the image using the Hough transform.
    If the detected circle is not roughly centered (based on a threshold), recenter it.
    Returns (center_y, center_x, radius) as integers.
    """
    # Convert to grayscale and blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 1)

    # Detect edges
    edges = cv2.Canny(gray_blurred, 10, 10)

    # Define radii range for circle detection (adjust as needed)
    hough_radii = np.arange(370, 420, 25)
    hough_res = hough_circle(edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    img_height, img_width = img.shape[:2]
    image_center_x = img_width // 2
    image_center_y = img_height // 2
    threshold = int(img_width * 0.15)  # allow 15% deviation

    if len(cx) > 0:
        center_y, center_x, radius = cy[0], cx[0], radii[0]
        # If the detected circle is not "more or less centered", recenter it.
        if abs(center_x - image_center_x) > threshold:
            center_x = image_center_x
            center_y = image_center_y
    else:
        # If no circle is detected, default to the image center and choose a radius
        center_x, center_y = image_center_x, image_center_y
        radius = min(image_center_x, image_center_y)

    # Adjust if the circle extends beyond the bottom edge
    if center_y + radius > img_height:
        center_y = img_height - radius

    return int(center_y), int(center_x), int(radius)

def create_circular_mask(h, w, center=None, radius=None):
    """
    Create a binary mask with a filled circle (True inside the circle, False outside)
    for an image of height h and width w.
    """
    if center is None:
        center = (int(w/2), int(h/2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask

def post_processing_first_step_2_classes(prediction_path, mask_path, images_folder):

    # Extract predictions and ground truth patches from H5 files
    predictions, ground_truth = extract_ordered_h5_testing(prediction_path, mask_path)
    # Get sorted list of original image file paths
    img_files = sorted(glob(os.path.join(images_folder, "*.png")))

    # Process each image patch: detect circle on the original image and apply a circular mask
    num_images = predictions.shape[0]
    processed_predictions = []
    processed_ground_truth = []

    for i in range(num_images):
        # Read the corresponding original image
        if i >= len(img_files):
            print(f"Not enough original images for index {i}")
            break
        img = cv2.imread(img_files[i], cv2.IMREAD_COLOR)
        if img is None:
            print(f"Could not read image: {img_files[i]}")
            continue

        # Desired output size
        #desired_h, desired_w = 1088, 1088
        desired_h, desired_w = 896, 896
        h, w = img.shape[:2]

        # Calculate padding
        top = (desired_h - h) // 2
        bottom = desired_h - h - top
        left = (desired_w - w) // 2
        right = desired_w - w - left

        # Apply padding to center the image with black borders
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Detect (or recenter) the circle from the original image
        center_y, center_x, radius = detect_centered_circle(img)

        # Assuming each prediction/ground truth patch has shape (channels, height, width)
        # Use the height and width from the first channel (assuming they are the same for all channels)
        _, h, w = predictions[i].shape
        # Create a binary circular mask (True inside the circle, False outside)
        circ_mask = create_circular_mask(h, w, center=(center_x, center_y), radius=radius)

        # Apply the mask to each channel. Multiplying by a boolean mask will set values outside the circle to 0.
        masked_prediction = predictions[i] * circ_mask
        # 2) Remap classes: gray (1) -> 90, white (2) -> 255
        masked_prediction[masked_prediction == 1] = 90
        masked_prediction[masked_prediction == 2] = 255

        # 3) Isolate white mask
        white_mask = (masked_prediction == 255)
        masked_ground_truth = np.where(ground_truth[i] >= 0.5, 1, 0).astype(np.uint8)

        processed_predictions.append(white_mask)
        processed_ground_truth.append(masked_ground_truth)

    # Convert lists back to numpy arrays
    processed_predictions = np.stack(processed_predictions, axis=0)
    processed_ground_truth = np.stack(processed_ground_truth, axis=0)

    print("Processed Predictions Shape:", processed_predictions.shape)
    print("Processed Ground Truth Shape:", processed_ground_truth.shape)

    return processed_predictions, processed_ground_truth

def post_processing_first_step_1_class(prediction_path, mask_path, images_folder):

    # Extract predictions and ground truth patches from H5 files
    predictions, ground_truth = extract_ordered_h5_testing(prediction_path, mask_path)
    # Get sorted list of original image file paths
    img_files = sorted(glob(os.path.join(images_folder, "*.png")))

    # Process each image patch: detect circle on the original image and apply a circular mask
    num_images = predictions.shape[0]
    processed_predictions = []
    processed_ground_truth = []

    for i in range(num_images):
        # Read the corresponding original image
        if i >= len(img_files):
            print(f"Not enough original images for index {i}")
            break
        img = cv2.imread(img_files[i], cv2.IMREAD_COLOR)
        if img is None:
            print(f"Could not read image: {img_files[i]}")
            continue

        # Desired output size
        #desired_h, desired_w = 1088, 1088
        desired_h, desired_w = 896, 896
        h, w = img.shape[:2]

        # Calculate padding
        top = (desired_h - h) // 2
        bottom = desired_h - h - top
        left = (desired_w - w) // 2
        right = desired_w - w - left

        # Apply padding to center the image with black borders
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Detect (or recenter) the circle from the original image
        center_y, center_x, radius = detect_centered_circle(img)

        # Assuming each prediction/ground truth patch has shape (channels, height, width)
        # Use the height and width from the first channel (assuming they are the same for all channels)
        _, h, w = predictions[i].shape
        # Create a binary circular mask (True inside the circle, False outside)
        circ_mask = create_circular_mask(h, w, center=(center_x, center_y), radius=radius)

        # Apply the mask to each channel. Multiplying by a boolean mask will set values outside the circle to 0.
        masked_prediction = predictions[i] * circ_mask
        masked_ground_truth = np.where(ground_truth[i] >= 0.5, 1, 0).astype(np.uint8)

        processed_predictions.append(masked_prediction)
        processed_ground_truth.append(masked_ground_truth)

    # Convert lists back to numpy arrays
    processed_predictions = np.stack(processed_predictions, axis=0)
    processed_ground_truth = np.stack(processed_ground_truth, axis=0)

    print("Processed Predictions Shape:", processed_predictions.shape)
    print("Processed Ground Truth Shape:", processed_ground_truth.shape)

    return processed_predictions, processed_ground_truth

'''
def expand_dirt_from_predictions(processed_predictions):
    # Define label values
    GRAY = 90  # Dirt / Gray class
    WHITE = 255  # White class

    updated_predictions = []

    # Loop over each prediction in your processed_predictions list
    for pred in processed_predictions:
        # Ensure pred is 2D (if it has a channel dimension, take the first channel)
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred_mask = pred[0].copy()
        elif pred.ndim == 2:
            pred_mask = pred.copy()
        else:
            pred_mask = pred[0].copy()

        img_remap = pred_mask.copy().astype(np.uint8)
        img_remap[pred_mask == 1] = 90
        img_remap[pred_mask == 2] = 255

        pred_mask = img_remap

        # Create a binary mask where white pixels are 1 (and others 0)
        white_mask = (pred_mask == WHITE).astype(np.uint8)

        # Compute a distance transform based on gray regions:
        # For the distance transform, we want gray pixels (value==GRAY) to act as "zero" points.
        # Create an image where pixels not equal to GRAY are set to 1.
        inv_gray = (pred_mask != GRAY).astype(np.uint8)
        # Compute the Euclidean distance transform.
        # Each pixel in 'dist' now represents the distance to the nearest gray pixel.
        dist = cv2.distanceTransform(inv_gray, cv2.DIST_L2, 3)

        # Label connected white regions (white components)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_mask, connectivity=8)

        # Loop over each white component (skip label 0, which is background)
        for label in range(1, num_labels):
            # Get the centroid (x, y) coordinates
            cx, cy = centroids[label]
            # Round coordinates to nearest integer pixel indices
            cx = int(round(cx))
            cy = int(round(cy))

            # Check if the distance at the centroid is 10 pixels or less
            if dist[cy, cx] <= 10:
                # Set all pixels in this connected component to GRAY
                pred_mask[labels == label] = GRAY

        updated_predictions.append(pred_mask.copy())

    # Optionally, convert updated_predictions back to a numpy array
    updated_predictions = np.stack(updated_predictions, axis=0)

    return updated_predictions
'''
def remove_70microns_particles(updated_predictions, median_px_mm=29.98):
    new_predictions = []

    for pred in updated_predictions:
        # 1) Flatten to a 2D uint8 mask
        if pred.ndim == 3 and pred.shape[0] == 1:
            mask = pred[0].astype(np.uint8)
        else:
            mask = pred.astype(np.uint8)

        '''
        # 2) Remap classes: gray (1) -> 90, white (2) -> 255
        mask[mask == 1] = 90
        mask[mask == 2] = 255

        # 3) Isolate white mask
        white_mask = (mask == 255)
        '''
        white_mask = mask
        # 4) Label and filter white blobs
        cleaned_white = np.zeros_like(white_mask, dtype=bool)
        lbl_white = label(white_mask, connectivity=1)
        for region in regionprops(lbl_white):
            major_mm = region.major_axis_length / median_px_mm
            if major_mm > 0.07:
                cleaned_white[lbl_white == region.label] = True

        # 5) The final mask keeps only the filtered white regions
        cleaned = cleaned_white.astype(np.uint8)
        new_predictions.append(cleaned)

    return np.stack(new_predictions, axis=0)

def store_h5(base_dir, new_predictions_array, new_ground_truths_array):
    save_target_path = os.path.join(base_dir, 'post_process_target.h5')
    save_predict_path = os.path.join(base_dir, 'post_process_predicted.h5')

    target = new_ground_truths_array
    predictions = new_predictions_array
    print(target.shape)
    print(predictions.shape)

    target_h5f = h5py.File(save_target_path, 'w')
    predicted_h5f = h5py.File(save_predict_path, 'w')

    total_number = len(target)
    for i in range(total_number):
        # print(i)
        target_data = target[i, ...]
        target_h5f.create_dataset(str(i), data=target_data)

        predicted_data = predictions[i, ...]
        predicted_h5f.create_dataset(str(i), data=predicted_data)
    target_h5f.close()
    predicted_h5f.close()

# --------------------------- Define Directories -----------------------

#Paths for post_processing_first_step
prediction_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Ensemble_v4\recon_predicted_ensemble_v4.h5"
mask_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Ensemble_v4\recon_target_ensemble_v4.h5"
images_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Focal_Loss\test\images"

#Base directory for store_h5
h5_directory = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Ensemble_v4\test"

#---------------------------- Run Script --------------------------------

#1. Post-process both predictions (Hough circle detection of Field of View) and masks (Binarization to remove "dirt" class)
processed_predictions, processed_ground_truth = post_processing_first_step_2_classes(prediction_path, mask_path, images_path)

'''
#2. Expand "dirt" class from predictions (Make really close white particles "dirt" if they are too close to "dirty area"
updated_predictions = expand_dirt_from_predictions(processed_predictions)
'''

#3. Remove 70 microns particles in both predictions and ground truth; remove "dirt" class from predictions through binarization
new_predictions_array = remove_70microns_particles(processed_predictions, median_px_mm=29.98)
new_ground_truths_array = remove_70microns_particles(processed_ground_truth, median_px_mm=29.98)

#4. Store post-processed predictions and ground truth masks in h5 format
store_h5(h5_directory, new_predictions_array, new_ground_truths_array)

#------------------------- Custom Visualization --------------------------------------------------------------------------

prediction_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Ensemble_v4\test\post_process_predicted.h5"
mask_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Ensemble_v4\test\post_process_target.h5"

prediction_images, mask_images = extract_ordered_h5_testing(prediction_path, mask_path)

num_samples = 2
# Set up the figure for visualization
fig, axes = plt.subplots(num_samples, 2, figsize=(10, 20))
for i in range(num_samples):
    # Plot original image
    axes[i, 0].imshow(prediction_images[i+50], cmap='gray')
    axes[i, 0].set_title(f"Predicted Image {i+50 + 1} with shape: {(prediction_images[i+50]).shape}")
    axes[i, 0].axis('off')

    # Plot mask
    axes[i, 1].imshow(mask_images[i+50], cmap='gray')
    axes[i, 1].set_title(f"GT Mask {i+50 + 1} with shape: {(prediction_images[i+50]).shape}")
    axes[i, 1].axis('off')
plt.show()
