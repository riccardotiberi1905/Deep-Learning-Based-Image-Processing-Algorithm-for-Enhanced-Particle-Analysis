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

def extract_large_gt_dice_df(pred_h5, gt_h5, median_px_mm=29.98):
    """
    For every GT particle ≥0.8 mm, merge all overlapping predicted components
    and compute its Dice against that single GT mask. Return a DataFrame:
      [image_idx, gt_label, length_mm, dice]
    """

    records = []
    for img_idx, (pred, gt) in enumerate(zip(preds, gts)):
        lbl_pred = label(pred, connectivity=1)
        lbl_gt = label(gt, connectivity=1)
        props_pred = regionprops(lbl_pred)
        # map label → regionprop for quick lookup
        pred_map = {rp.label: rp for rp in props_pred}

        for rg in regionprops(lbl_gt):
            length_mm = rg.major_axis_length / median_px_mm
            if length_mm < 0.8:
                continue

            gt_mask = (lbl_gt == rg.label)
            overlapping_labels = np.unique(lbl_pred[gt_mask])
            overlapping_labels = overlapping_labels[overlapping_labels != 0]

            # sizes of each overlapping pred in mm
            pred_sizes = [
                round(pred_map[lab].major_axis_length / median_px_mm, 3)
                for lab in overlapping_labels
            ]

            n_overlaps = len(pred_sizes)

            # merged-pred mask → Dice
            if n_overlaps == 0:
                dice = 0.0
            else:
                merged_pred = np.isin(lbl_pred, overlapping_labels).astype(np.uint8)
                inter = np.sum(merged_pred * gt_mask.astype(np.uint8))
                dice = 2 * inter / (np.sum(merged_pred) + np.sum(gt_mask) + 1e-8)

            records.append({
                'image_idx': img_idx,
                'gt_label': rg.label,
                'length_mm': round(length_mm, 3),
                'n_overlaps': n_overlaps,
                'Pred_particle_sizes': pred_sizes,
                'dice': round(dice, 6)
            })

    df = pd.DataFrame.from_records(records,
                                   columns=['image_idx', 'gt_label', 'length_mm', 'n_overlaps', 'Pred_particle_sizes',
                                            'dice'])
    return df
#---------------------- Define Directories
prediction_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Ensemble_v4\test_1\post_process_predicted.h5"
mask_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Ensemble_v4\test_1\post_process_target.h5"
output_dir = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Ensemble_v4\test_1\Results"
#---------------------- Run Script -----------------------------------

#1. Read h5 files
new_predictions_array, new_ground_truths_array = extract_ordered_h5_testing(prediction_path, mask_path)

# 1. Load your patches
preds, gts = extract_ordered_h5_testing(prediction_path, mask_path)

# 2. Evaluate
df_dice = extract_large_gt_dice_df(preds, gts)


excel_name_1 = 'mod_dice_per_image.xlsx'

save_path_1 = os.path.join(output_dir, excel_name_1)

#Store in excel
df_dice.to_excel(save_path_1)


