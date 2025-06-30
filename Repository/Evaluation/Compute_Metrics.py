import h5py
import os
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

def dice_coefficient(pred, gt):
    """
    Compute Dice coefficient for the entire mask.
    """
    intersection = np.sum(pred * gt)
    return (2. * intersection) / (np.sum(pred) + np.sum(gt) + 1e-8)  # Avoid division by zero


def dice_foreground(pred, gt):
    """
    Compute Dice for the foreground (object).
    """
    pred_fg = (pred == 1)
    gt_fg = (gt == 1)
    return dice_coefficient(pred_fg, gt_fg)


def dice_background(pred, gt):
    """
    Compute Dice for the background.
    """
    pred_bg = (pred == 0)
    gt_bg = (gt == 0)
    return dice_coefficient(pred_bg, gt_bg)


def calculate_iou(binary_image1, binary_image2):
    binary_image1 = binary_image1.astype(bool)  # Convert to boolean
    binary_image2 = binary_image2.astype(bool)  # Convert to boolean

    intersection = np.sum(binary_image1 & binary_image2)
    union = np.sum(binary_image1 | binary_image2)

    IoU = intersection / (union + 1e-8)

    return IoU

def safe_mean(arr):
    """
    Return the mean of a list; if empty, return np.nan.
    """
    return np.mean(arr) if len(arr) > 0 else np.nan

#---------------------- Define Directories
prediction_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Ensemble_v4\test_1\post_process_predicted.h5"
mask_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Ensemble_v4\test_1\post_process_target.h5"
output_dir = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Ensemble_v4\test_1\Results"
#---------------------- Run Script -----------------------------------

#1. Read h5 files
new_predictions_array, new_ground_truths_array = extract_ordered_h5_testing(prediction_path, mask_path)

# Initialize an empty list to store individual rows of metrics
median_px_mm = 29.98 #Pixel -- mm relationship
threshold = 10  # Pixel distance threshold for matching
metrics_list = []

for index in range(len(new_predictions_array)):
    # Compute pixel-level metrics
    overall_dice = dice_coefficient(new_predictions_array[index], new_ground_truths_array[index])
    foreground_dice = dice_foreground(new_predictions_array[index], new_ground_truths_array[index])
    background_dice = dice_background(new_predictions_array[index], new_ground_truths_array[index])
    overall_iou = calculate_iou(new_predictions_array[index], new_ground_truths_array[index])
    foreground_iou = calculate_iou(new_predictions_array[index] == 1, new_ground_truths_array[index] == 1)
    background_iou = calculate_iou(new_predictions_array[index] == 0, new_ground_truths_array[index] == 0)

    # Initialize lists for different particle sizes for GROUND TRUTH
    particles_small = []  # 70-300 microns
    particles_medium = []  # >300-600 microns
    particles_large = []  # >600-1000 microns
    particles_extra_large = []  # >1000 microns

    # Initialize lists for different particle sizes for PREDICTIONS
    pred_small = []  # 70-300 microns
    pred_medium = []  # >300-600 microns
    pred_large = []  # >600-1000 microns
    pred_extra_large = []  # >1000 microns

    # Presence of particles of certain size
    presence_small = True
    presence_medium = True
    presence_large = True
    presence_extra_large = True

    #Initialize total area for predictions and groundtruths
    area_pred = 0
    area_gt = 0

    # Label connected components
    label_pred = label(new_predictions_array[index], connectivity=1)
    particles_pred = regionprops(label_pred)
    total_pred = len(particles_pred)
    for particle in particles_pred:
        area = particle.area
        area = np.round((area / (median_px_mm ** 2)), 3)
        area_pred += area
        major_axis_length = particle.major_axis_length
        major_axis_length = np.round((major_axis_length / median_px_mm), 3)
        if major_axis_length >= 1:
            pred_extra_large.append(major_axis_length)
        elif major_axis_length >= 0.6:
            pred_large.append(major_axis_length)
        elif major_axis_length >= 0.3:
            pred_medium.append(major_axis_length)
        else:
            pred_small.append(major_axis_length)

    label_mask = label(new_ground_truths_array[index], connectivity=1)
    particles_mask = regionprops(label_mask)
    total_mask = len(particles_mask)
    for particle in particles_mask:
        area = particle.area
        area = np.round((area / (median_px_mm ** 2)), 3)
        area_gt += area
        major_axis_length = particle.major_axis_length
        major_axis_length = np.round((major_axis_length / median_px_mm), 3)
        if major_axis_length >= 1:
            particles_extra_large.append(major_axis_length)
        elif major_axis_length >= 0.6:
            particles_large.append(major_axis_length)
        elif major_axis_length >= 0.3:
            particles_medium.append(major_axis_length)
        else:
            particles_small.append(major_axis_length)


    if len(particles_small) > 0:
        presence_small = True
    else:
        presence_small = False

    if len(particles_medium) > 0:
        presence_medium = True
    else:
        presence_medium = False

    if len(particles_large) > 0:
        presence_large = True
    else:
        presence_large = False

    if len(particles_extra_large) > 0:
        presence_extra_large = True
    else:
        presence_extra_large = False


    # Evaluate particle matching when both masks have particles
    particle_matches = 0  # True Positives
    unmatched_gts = 0  # False Negatives

    particle_matches_small = 0
    particle_matches_medium = 0
    particle_matches_large = 0
    particle_matches_extra_large = 0

    unmatched_gts_small = 0
    unmatched_gts_medium = 0
    unmatched_gts_large = 0
    unmatched_gts_extra_large = 0

    # Initialize lists for different particle sizes to store differences in Major Axis Length for that specific size range
    diff_particles_small = []  # 70-300 microns
    diff_particles_medium = []  # >300-600 microns
    diff_particles_large = []  # >600-1000 microns
    diff_particles_extra_large = []  # >1000 microns

    # Handle edge cases when one of the masks has no particles
    if total_mask == 0 and total_pred > 0:
        metrics_list.append({
            'Index': index,
            'IoU_White': 0,
            'IoU_Black': background_iou,
            'Dice_White': 0,
            'Dice_Black': background_dice,
            'Overall_Dice': overall_dice,
            'Overall_IoU': overall_iou,
            'Total_Particles_Pred': total_pred,
            'Total_Particles_GT': total_mask,
            'True_Positive_Particles': 0,
            'False_Negative_Particles': 0,
            'False_Positive_Particles': total_pred,
            'Mean_Length_Difference_Small': 0,
            'True_Positive_Particles_Small': 0,
            'False_Negative_Particles_Small': 0,
            'False_Positive_Particles_Small': len(pred_small),
            'Mean_Length_Difference_Medium': 0,
            'True_Positive_Particles_Medium': 0,
            'False_Negative_Particles_Medium': 0,
            'False_Positive_Particles_Medium': len(pred_medium),
            'Mean_Length_Difference_Large': 0,
            'True_Positive_Particles_Large': 0,
            'False_Negative_Particles_Large': 0,
            'False_Positive_Particles_Large': len(pred_large),
            'Mean_Length_Difference_Extra_Large': 0,
            'True_Positive_Particles_Extra_Large': 0,
            'False_Negative_Particles_Extra_Large': 0,
            'False_Positive_Particles_Extra_Large': len(pred_extra_large),
            'Total_Area_Prediction': area_pred,
            'Total_Area_GT': area_gt,
            'Total_Area_Difference': np.abs(area_pred-area_gt),
            'Presence_Small': False,
            'Presence_Medium': False,
            'Presence_Large': False,
            'Presence_Extra_Large': False,
            'Total_Pred_Small': len(pred_small),
            'Total_Pred_Medium': len(pred_medium),
            'Total_Pred_Large': len(pred_large),
            'Total_Pred_Extra_Large': len(pred_extra_large)
        })
        continue  # Move to next image

    if total_pred == 0 and total_mask > 0:
        metrics_list.append({
            'Index': index,
            'IoU_White': 0,
            'IoU_Black': background_iou,
            'Dice_White': 0,
            'Dice_Black': background_dice,
            'Overall_Dice': overall_dice,
            'Overall_IoU': overall_iou,
            'Total_Particles_Pred': total_pred,
            'Total_Particles_GT': total_mask,
            'True_Positive_Particles': 0,
            'False_Negative_Particles': total_mask,
            'False_Positive_Particles': 0,
            'True_Positive_Particles_Small': 0,
            'Mean_Length_Difference_Small': 0,
            'False_Negative_Particles_Small': len(particles_small),
            'False_Positive_Particles_Small': 0,
            'Mean_Length_Difference_Medium': 0,
            'True_Positive_Particles_Medium': 0,
            'False_Negative_Particles_Medium': len(particles_medium),
            'False_Positive_Particles_Medium': 0,
            'Mean_Length_Difference_Large': 0,
            'True_Positive_Particles_Large': 0,
            'False_Negative_Particles_Large': len(particles_large),
            'False_Positive_Particles_Large': 0,
            'Mean_Length_Difference_Extra_Large': 0,
            'True_Positive_Particles_Extra_Large': 0,
            'False_Negative_Particles_Extra_Large': len(particles_extra_large),
            'False_Positive_Particles_Extra_Large': 0,
            'Total_Area_Prediction': area_pred,
            'Total_Area_GT': area_gt,
            'Total_Area_Difference': np.abs(area_pred - area_gt),
            'Presence_Small': presence_small,
            'Presence_Medium': presence_medium,
            'Presence_Large': presence_large,
            'Presence_Extra_Large': presence_extra_large,
            'Total_Pred_Small': len(pred_small),
            'Total_Pred_Medium': len(pred_medium),
            'Total_Pred_Large': len(pred_large),
            'Total_Pred_Extra_Large': len(pred_extra_large)
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
            unmatched_gts += 1  # No match found; count as a false negative
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
            particle_matches += 1
            length_gt = np.round((gt_region.major_axis_length/ median_px_mm), 3)
            length_pred = np.round((best_match.major_axis_length / median_px_mm), 3)
            length_difference = np.abs(length_gt - length_pred)
            if length_gt >= 1:
                diff_particles_extra_large.append(length_difference)
                particle_matches_extra_large += 1
            elif length_gt >= 0.6:
                diff_particles_large.append(length_difference)
                particle_matches_large += 1
            elif length_gt >= 0.3:
                diff_particles_medium.append(length_difference)
                particle_matches_medium += 1
            else:
                diff_particles_small.append(length_difference)
                particle_matches_small += 1
        else:
            unmatched_gts += 1
            length_gt = np.round((gt_region.major_axis_length / median_px_mm), 3)
            if length_gt >= 1:
                unmatched_gts_extra_large += 1
            elif length_gt >= 0.6:
                unmatched_gts_large += 1
            elif length_gt >= 0.3:
                unmatched_gts_medium += 1
            else:
                unmatched_gts_small += 1

    # False Positives: predicted particles that were not matched
    unmatched_pred = total_pred - particle_matches
    unmatched_pred_small = len(pred_small) - particle_matches_small
    unmatched_pred_medium = len(pred_medium) - particle_matches_medium
    unmatched_pred_large = len(pred_large) - particle_matches_large
    unmatched_pred_extra_large = len(pred_extra_large) - particle_matches_extra_large

    # Append current metrics as a dictionary for this image
    metrics_list.append({
        'Index': index,
        'IoU_White': foreground_iou,
        'IoU_Black': background_iou,
        'Dice_White': foreground_dice,
        'Dice_Black': background_dice,
        'Overall_Dice': overall_dice,
        'Overall_IoU': overall_iou,
        'Total_Particles_Pred': total_pred,
        'Total_Particles_GT': total_mask,
        'True_Positive_Particles': particle_matches,
        'False_Negative_Particles': unmatched_gts,
        'False_Positive_Particles': unmatched_pred,
        'Mean_Length_Difference_Small': np.round(safe_mean(diff_particles_small), 3),
        'True_Positive_Particles_Small': particle_matches_small,
        'False_Negative_Particles_Small': unmatched_gts_small,
        'False_Positive_Particles_Small': unmatched_pred_small,
        'Mean_Length_Difference_Medium': np.round(safe_mean(diff_particles_medium), 3),
        'True_Positive_Particles_Medium': particle_matches_medium,
        'False_Negative_Particles_Medium': unmatched_gts_medium,
        'False_Positive_Particles_Medium': unmatched_pred_medium,
        'Mean_Length_Difference_Large': np.round(safe_mean(diff_particles_large), 3),
        'True_Positive_Particles_Large': particle_matches_large,
        'False_Negative_Particles_Large': unmatched_gts_large,
        'False_Positive_Particles_Large': unmatched_pred_large,
        'Mean_Length_Difference_Extra_Large': np.round(safe_mean(diff_particles_extra_large), 3),
        'True_Positive_Particles_Extra_Large': particle_matches_extra_large,
        'False_Negative_Particles_Extra_Large': unmatched_gts_extra_large,
        'False_Positive_Particles_Extra_Large': unmatched_pred_extra_large,
        'Total_Area_Prediction': area_pred,
        'Total_Area_GT': area_gt,
        'Total_Area_Difference': np.abs(area_pred - area_gt),
        'Presence_Small': presence_small,
        'Presence_Medium': presence_medium,
        'Presence_Large': presence_large,
        'Presence_Extra_Large': presence_extra_large,
        'Total_Pred_Small': len(pred_small),
        'Total_Pred_Medium': len(pred_medium),
        'Total_Pred_Large': len(pred_large),
        'Total_Pred_Extra_Large': len(pred_extra_large)
    })


metrics_path = os.path.join(output_dir, "General_Metrics.xlsx")
metrics = pd.DataFrame(metrics_list)

#Store in excel
metrics.to_excel(metrics_path)

print(f"Saved Metrics excel to {metrics_path}")
