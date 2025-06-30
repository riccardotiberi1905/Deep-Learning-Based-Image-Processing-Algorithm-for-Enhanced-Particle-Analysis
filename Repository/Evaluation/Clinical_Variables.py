import os.path

import h5py
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt


#----------------------- Functions --------------------------------------
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

def regionprops_particle(particles, px_mm):
    total_area = 0
    area_length = []
    part_1_mm = 0
    for part in particles:
        area = part.area
        area = np.round((area / (px_mm ** 2)), 3)
        total_area += area
        major_axis_length = part.major_axis_length
        major_axis_length = np.round((major_axis_length / median_px_mm), 3)
        area_length.append((major_axis_length, area))
        if major_axis_length >= 1:
            part_1_mm += 1

    return total_area, area_length, part_1_mm

def max_size_area(area_length):

    if not area_length:
        max_size = 0
        max_area = 0
    else:
        max_item = max(area_length, key=lambda x: x[0])
        max_size = max_item[0]
        max_area = max_item[1]

    return max_size, max_area


#---------------------- Define Directories----------------------------
prediction_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\2_classes\test_1\post_process_predicted.h5"
ground_truth_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\2_classes\test_1\post_process_target.h5"
output_dir = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\2_classes\test_1\Results"

#---------------------- Run Script -----------------------------------
#1. Read h5 files
new_predictions_array, new_ground_truths_array = extract_ordered_h5_testing(prediction_path, ground_truth_path)

# Initialize an empty list to store individual rows of metrics
median_px_mm = 29.98 #Pixel -- mm relationship
metrics_list = []
indexes_to_plot = []

for index in range(len(new_predictions_array)):

    pred_bin = new_predictions_array[index]
    gt_bin = new_ground_truths_array[index]

    # Label connected components
    #Prediction
    label_pred = label(pred_bin, connectivity=1)
    particles_pred = regionprops(label_pred)
    total_pred = len(particles_pred)
    #Ground Truth
    label_gt = label(gt_bin, connectivity=1)
    particles_gt = regionprops(label_gt)
    total_gt = len(particles_gt)

    #Iterate over particles
    #Prediction
    area_pred, area_pred_length, pred_1_mm = regionprops_particle(particles_pred, median_px_mm)
    #Ground Truth
    area_gt, area_gt_length, gt_1_mm = regionprops_particle(particles_gt, median_px_mm)

    #Max Size and Max Area
    # Prediction
    max_size_pred, max_area_pred = max_size_area(area_pred_length)
    # Ground Truth
    max_size_gt, max_area_gt = max_size_area(area_gt_length)

    Total_Count_Diff = np.abs(total_pred-total_gt)
    Max_Size_Diff = np.abs(max_size_pred - max_size_gt)
    Max_Area_Diff = np.abs(max_area_pred - max_area_gt)
    Particles_1mm_Diff = np.abs(pred_1_mm - gt_1_mm)

    if Particles_1mm_Diff > 0:
        indexes_to_plot.append(index)

    metrics_list.append({
        'Index': index,
        'Total_Count_Pred': total_pred,
        'Max_Size_Pred': max_size_pred,
        'Max_Area_Pred': max_area_pred,
        'Total_Area_Pred': area_pred,
        'Particles_>1mm_Pred': pred_1_mm,
        'Total_Count_GT': total_gt,
        'Max_Size_GT': max_size_gt,
        'Max_Area_GT': max_area_gt,
        'Total_Area_GT': area_gt,
        'Particles_>1mm_GT': gt_1_mm,
        'Total_Count_Diff': Total_Count_Diff,
        'Max_Size_Diff': Max_Size_Diff,
        'Max_Area_Diff': Max_Area_Diff,
        'Particles_>1mm_Diff': Particles_1mm_Diff
    })

metrics = pd.DataFrame(metrics_list)

excel_name = 'clinical_variables.xlsx'

save_path = os.path.join(output_dir, excel_name)
#Store in excel
metrics.to_excel(save_path)

'''
# 1) Set up sub-directories under your existing output_dir
pred_folder = os.path.join(output_dir, "predictions")
gt_folder   = os.path.join(output_dir, "ground_truths")
os.makedirs(pred_folder, exist_ok=True)
os.makedirs(gt_folder,   exist_ok=True)

# 2) Loop over every index you want to save
for idx in indexes_to_plot:
    pred_img = new_predictions_array[idx]      # shape: (C, H, W) or (H, W)
    gt_img   = new_ground_truths_array[idx]    # same

    # If you have multi-channel, pick the channel or collapse:
    # pred_img = pred_img[0]  # for example

    # 3) Write them out as PNGs named “<index>.png”
    pred_path = os.path.join(pred_folder, f"{idx}.png")
    gt_path   = os.path.join(gt_folder,   f"{idx}.png")

    # Use matplotlib’s imsave (handles grayscale arrays nicely)
    plt.imsave(pred_path, pred_img, cmap="gray")
    plt.imsave(gt_path,   gt_img,   cmap="gray")
'''