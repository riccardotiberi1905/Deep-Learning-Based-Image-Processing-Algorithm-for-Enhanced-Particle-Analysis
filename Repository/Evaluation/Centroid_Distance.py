import h5py
import os
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops

#------------------------- Essential Functions -------------------------------
def extract_ordered_h5_testing(images_path, mask_path):
    """
    Extract ordered image and mask patches from h5 files for further reconstruction
    """
    with h5py.File(images_path, 'r') as f:
        keys = sorted(list(f.keys()), key=lambda x: int(x))
        samples = [f[key][...] for key in keys]
        image_patches = np.stack(samples, axis=0)
    with h5py.File(mask_path, 'r') as f:
        keys = sorted(list(f.keys()), key=lambda x: int(x))
        samples = [f[key][...] for key in keys]
        mask_patches = np.stack(samples, axis=0)
    return image_patches, mask_patches

def dice_coefficient(pred, gt):
    """
    Compute Dice coefficient for the entire mask.
    """
    intersection = np.sum(pred * gt)
    return (2. * intersection) / (np.sum(pred) + np.sum(gt) + 1e-8)  # Avoid division by zero

#---------------------- Define Directories and Parameters --------------------
prediction_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Focal_Loss\test\post_process_predicted.h5"
mask_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Focal_Loss\test\post_process_target.h5"
output_dir = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Focal_Loss\test\Results"
os.makedirs(output_dir, exist_ok=True)

# Pixel-to-mm conversion
median_px_mm = 29.98

# Define size bins (mm)
edges = [0, 0.07, 0.1] + list(np.arange(0.2, 2.1, 0.1)) + [np.inf]
labels = [f'{lo:.2f}-{hi:.2f}' if hi < np.inf else f'>{lo:.2f}' for lo, hi in zip(edges[:-1], edges[1:])]

#---------------------- Load Data -------------------------------------------
images, masks = extract_ordered_h5_testing(prediction_path, mask_path)

# Prepare storage for overlap pairs
overlap_data = { (gt_label, pr_label): [] for gt_label in labels for pr_label in labels }

#---------------------- Analyze Overlaps ------------------------------------
for idx in range(len(images)):
    gt_map = masks[idx].astype(bool)
    pr_map = images[idx].astype(bool)

    # Label connected components
    gt_labels = label(gt_map, connectivity=1)
    pr_labels = label(pr_map, connectivity=1)

    particles_gt = regionprops(gt_labels)
    particles_pr = regionprops(pr_labels)

    for gt in particles_gt:
        gt_mask = (gt_labels == gt.label).astype(int)
        centroid_gt = np.array(gt.centroid)
        length_gt = np.round(gt.major_axis_length / median_px_mm, 3)
        gt_bin = labels[np.digitize(length_gt, edges) - 1]

        for pr in particles_pr:
            pr_mask = (pr_labels == pr.label).astype(int)
            # Compute Dice coefficient using the new function
            dice = dice_coefficient(pr_mask, gt_mask)
            if dice == 0:
                continue

            centroid_pr = np.array(pr.centroid)
            length_pr = np.round(pr.major_axis_length / median_px_mm, 3)
            pr_bin = labels[np.digitize(length_pr, edges) - 1]
            dist = np.linalg.norm(centroid_gt - centroid_pr)
            overlap_data[(gt_bin, pr_bin)].append({
                'ImageIndex': idx,
                'Dice': dice,
                'Distance_px': dist,
                'GT_Length_mm': length_gt,
                'PR_Length_mm': length_pr
            })

#---------------------- Build Summary Matrices ------------------------------
summary_mean = pd.DataFrame(index=labels, columns=labels, dtype=float)
summary_median = pd.DataFrame(index=labels, columns=labels, dtype=float)
summary_count = pd.DataFrame(index=labels, columns=labels, dtype=int)

for (gt_label, pr_label), records in overlap_data.items():
    distances = [r['Distance_px'] for r in records]
    summary_count.loc[gt_label, pr_label] = len(distances)
    summary_mean.loc[gt_label, pr_label] = np.nanmean(distances)
    summary_median.loc[gt_label, pr_label] = np.nanmedian(distances)

#---------------------- Save to Excel ---------------------------------------
with pd.ExcelWriter(os.path.join(output_dir, 'overlap_analysis.xlsx')) as wr:
    summary_count.to_excel(wr, sheet_name='Count')
    summary_mean.to_excel(wr, sheet_name='Mean_Distance')
    summary_median.to_excel(wr, sheet_name='Median_Distance')
    # detailed sheets per bin pair
    for (gt_label, pr_label), records in overlap_data.items():
        if not records:
            continue
        df = pd.DataFrame(records)
        sheet = f"GT_{gt_label}_PR_{pr_label}"[:31]
        df.to_excel(wr, sheet_name=sheet, index=False)

print("Saved overlap analysis Excel with summary and detailed sheets.")
