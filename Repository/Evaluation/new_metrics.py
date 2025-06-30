import h5py
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops

#------------------------- Essential Functions -------------------------------
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

def safe_mean(arr):
    """
    Return the mean of a list; if empty, return np.nan.
    """
    return np.mean(arr) if len(arr) > 0 else np.nan

#---------------------- Define Directories
prediction_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\2_classes\test\post_process_predicted.h5"
mask_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\2_classes\test\post_process_target.h5"

#---------------------- Run Script -----------------------------------

#1. Read h5 files
new_predictions_array, new_ground_truths_array = extract_ordered_h5_testing(prediction_path, mask_path)

# Initialize an empty list to store individual rows of metrics
median_px_mm = 29.98 #Pixel -- mm relationship
metrics_list = []

for index in range(len(new_predictions_array)):

    # Initialize lists for different particle sizes for GROUND TRUTH
    particles_70 = []  # 70-100 microns
    particles_100 = []  # >100-200 microns
    particles_200 = []  # >200-300 microns
    particles_300 = []  # >300-400 microns
    particles_400 = []  # >400-500 microns
    particles_500 = []  # >500-600 microns
    particles_600 = []  # >600-700 microns
    particles_700 = []  # >700-800 microns
    particles_800 = []  # >800-900 microns
    particles_900 = []  # >900-1000 microns
    particles_1000 = []  # >1000-1100 microns
    particles_1100 = []  # >1100-1200 microns
    particles_1200 = []  # >1200-1300 microns
    particles_1300 = []  # >1300-1400 microns
    particles_1400 = []  # >1400-1500 microns
    particles_1500 = []  # >1500-1600 microns
    particles_1600 = []  # >1600-1700 microns
    particles_1700 = []  # >1700-1800 microns
    particles_1800 = []  # >1800-1900 microns
    particles_1900 = []  # >1900-2000 microns
    particles_2000 = []  # >2000 microns

    # Initialize lists for different particle sizes for GROUND TRUTH
    pred_70 = []  # 70-100 microns
    pred_100 = []  # >100-200 microns
    pred_200 = []  # >200-300 microns
    pred_300 = []  # >300-400 microns
    pred_400 = []  # >400-500 microns
    pred_500 = []  # >500-600 microns
    pred_600 = []  # >600-700 microns
    pred_700 = []  # >700-800 microns
    pred_800 = []  # >800-900 microns
    pred_900 = []  # >900-1000 microns
    pred_1000 = []  # >1000-1100 microns
    pred_1100 = []  # >1100-1200 microns
    pred_1200 = []  # >1200-1300 microns
    pred_1300 = []  # >1300-1400 microns
    pred_1400 = []  # >1400-1500 microns
    pred_1500 = []  # >1500-1600 microns
    pred_1600 = []  # >1600-1700 microns
    pred_1700 = []  # >1700-1800 microns
    pred_1800 = []  # >1800-1900 microns
    pred_1900 = []  # >1900-2000 microns
    pred_2000 = []  # >2000 microns

    # Evaluate particle matching when both masks have particles
    threshold = 10  # Pixel distance threshold for matching

    # Label connected components
    label_pred = label(new_predictions_array[index], connectivity=1)
    particles_pred = regionprops(label_pred)
    total_pred = len(particles_pred)

    label_mask = label(new_ground_truths_array[index], connectivity=1)
    particles_mask = regionprops(label_mask)
    total_mask = len(particles_mask)

    # Handle edge cases when one of the masks has no particles
    if total_mask == 0 and total_pred > 0:
        metrics_list.append({
            'Index': index,
            'Total_Particles_Pred': total_pred,
            'Total_Particles_GT': total_mask,
            'Length_Difference_70': [],
            'Length_Difference_100': [],
            'Length_Difference_200': [],
            'Length_Difference_300': [],
            'Length_Difference_400': [],
            'Length_Difference_500': [],
            'Length_Difference_600': [],
            'Length_Difference_700': [],
            'Length_Difference_800': [],
            'Length_Difference_900': [],
            'Length_Difference_1000': [],
            'Length_Difference_1100': [],
            'Length_Difference_1200': [],
            'Length_Difference_1300': [],
            'Length_Difference_1400': [],
            'Length_Difference_1500': [],
            'Length_Difference_1600': [],
            'Length_Difference_1700': [],
            'Length_Difference_1800': [],
            'Length_Difference_1900': [],
            'Length_Difference_2000': []
        })
        continue  # Move to next image

    if total_pred == 0 and total_mask > 0:
        metrics_list.append({
            'Index': index,
            'Total_Particles_Pred': total_pred,
            'Total_Particles_GT': total_mask,
            'Length_Difference_70': [],
            'Length_Difference_100': [],
            'Length_Difference_200': [],
            'Length_Difference_300': [],
            'Length_Difference_400': [],
            'Length_Difference_500': [],
            'Length_Difference_600': [],
            'Length_Difference_700': [],
            'Length_Difference_800': [],
            'Length_Difference_900': [],
            'Length_Difference_1000': [],
            'Length_Difference_1100': [],
            'Length_Difference_1200': [],
            'Length_Difference_1300': [],
            'Length_Difference_1400': [],
            'Length_Difference_1500': [],
            'Length_Difference_1600': [],
            'Length_Difference_1700': [],
            'Length_Difference_1800': [],
            'Length_Difference_1900': [],
            'Length_Difference_2000': []
        })
        continue  # Move to next image


    # For each ground truth particle, try to find a matching predicted particle
    for gt_idx, gt_region in enumerate(particles_mask):
        centroid_gt = np.array(gt_region.centroid)

        # Compute distances from this GT particle to all predicted particles
        distances = [
            np.linalg.norm(centroid_gt - np.array(pred_region.centroid))
            for pred_region in particles_pred
        ]

        # Find predicted regions within the threshold distance
        matches = [i for i, d in enumerate(distances) if d < threshold]

        if not matches:
            continue

        # Select the closest match
        best_match_idx = matches[np.argmin([distances[i] for i in matches])]
        best_match = particles_pred[best_match_idx]  # Use consistent variable name

        # Ensure bidirectional matching: check that this predicted region's closest GT is the current one
        centroid_pred = np.array(best_match.centroid)
        reverse_distances = [
            np.linalg.norm(centroid_pred - np.array(other_gt.centroid))
            for other_gt in particles_mask
        ]
        reverse_matches = [i for i, d in enumerate(reverse_distances) if d < threshold]
        reverse_best_match_idx = reverse_matches[np.argmin([reverse_distances[i] for i in reverse_matches])]

        if reverse_best_match_idx == gt_idx:
            length_gt = np.round((gt_region.major_axis_length/ median_px_mm), 3)
            length_pred = np.round((best_match.major_axis_length / median_px_mm), 3)
            length_difference = np.abs(length_gt - length_pred)
            if length_gt >= 2:
                particles_2000.append(length_difference)
            elif length_gt >= 1.9:
                particles_1900.append(length_difference)
            elif length_gt >= 1.8:
                particles_1800.append(length_difference)
            elif length_gt >= 1.7:
                particles_1700.append(length_difference)
            elif length_gt >= 1.6:
                particles_1600.append(length_difference)
            elif length_gt >= 1.5:
                particles_1500.append(length_difference)
            elif length_gt >= 1.4:
                particles_1400.append(length_difference)
            elif length_gt >= 1.3:
                particles_1300.append(length_difference)
            elif length_gt >= 1.2:
                particles_1200.append(length_difference)
            elif length_gt >= 1.1:
                particles_1100.append(length_difference)
            elif length_gt >= 1.0:
                particles_1000.append(length_difference)
            elif length_gt >= 0.9:
                particles_900.append(length_difference)
            elif length_gt >= 0.8:
                particles_800.append(length_difference)
            elif length_gt >= 0.7:
                particles_700.append(length_difference)
            elif length_gt >= 0.6:
                particles_600.append(length_difference)
            elif length_gt >= 0.5:
                particles_500.append(length_difference)
            elif length_gt >= 0.4:
                particles_400.append(length_difference)
            elif length_gt >= 0.3:
                particles_300.append(length_difference)
            elif length_gt >= 0.2:
                particles_200.append(length_difference)
            elif length_gt >= 0.1:
                particles_100.append(length_difference)
            elif length_gt >= 0.07:
                particles_70.append(length_difference)

        else:
             continue

    # Append current metrics as a dictionary for this image
    metrics_list.append({
        'Index': index,
            'Total_Particles_Pred': total_pred,
            'Total_Particles_GT': total_mask,
            'Length_Difference_70': particles_70,
            'Length_Difference_100': particles_100,
            'Length_Difference_200': particles_200,
            'Length_Difference_300': particles_300,
            'Length_Difference_400': particles_400,
            'Length_Difference_500': particles_500,
            'Length_Difference_600': particles_600,
            'Length_Difference_700': particles_700,
            'Length_Difference_800': particles_800,
            'Length_Difference_900': particles_900,
            'Length_Difference_1000': particles_1000,
            'Length_Difference_1100': particles_1100,
            'Length_Difference_1200': particles_1200,
            'Length_Difference_1300': particles_1300,
            'Length_Difference_1400': particles_1400,
            'Length_Difference_1500': particles_1500,
            'Length_Difference_1600': particles_1600,
            'Length_Difference_1700': particles_1700,
            'Length_Difference_1800': particles_1800,
            'Length_Difference_1900': particles_1900,
            'Length_Difference_2000': particles_2000
    })


metrics = pd.DataFrame(metrics_list)

#Store in excel
metrics.to_excel("gt_vs_predictions_length_diff.xlsx")
