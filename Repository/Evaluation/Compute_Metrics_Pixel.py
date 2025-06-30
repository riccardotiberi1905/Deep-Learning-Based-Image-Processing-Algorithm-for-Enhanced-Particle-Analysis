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

def safe_mean(arr):
    """
    Return the mean of a list; if empty, return np.nan.
    """
    return np.mean(arr) if len(arr) > 0 else np.nan

#---------------------- Define Directories
prediction_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\2_classes\test_1\post_process_predicted.h5"
mask_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\2_classes\test_1\post_process_target.h5"
output_dir = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\2_classes\test_1\Results"
#---------------------- Run Script -----------------------------------

#1. Read h5 files
new_predictions_array, new_ground_truths_array = extract_ordered_h5_testing(prediction_path, mask_path)

# Initialize an empty list to store individual rows of metrics

metrics_list = []

for index in range(len(new_predictions_array)):

    pred_bin = new_predictions_array[index]
    gt_bin = new_ground_truths_array[index]

    # counts of positives in each
    n_pred = int(pred_bin.sum())
    n_gt = int(gt_bin.sum())

    # handle empty‐mask edge cases
    if n_pred == 0 and n_gt == 0:
        # nothing to compare
        tp_px = fp_px = fn_px = 0
    elif n_pred == 0:
        # no predicted positives → all GT positives are FN
        tp_px = 0
        fp_px = 0
        fn_px = n_gt
    elif n_gt == 0:
        # no GT positives → all preds are FP
        tp_px = 0
        fp_px = n_pred
        fn_px = 0
    else:
        # normal case
        # 2) count the overlap
        tp_px = int(np.count_nonzero(pred_bin & gt_bin))
        # 3) derive the rest by subtraction
        fp_px = int(pred_bin.sum()) - tp_px
        fn_px = int(gt_bin.sum()) - tp_px


    # append row
    metrics_list.append({
        'patch_index': index,
        'n_pred_px': n_pred,
        'n_gt_px': n_gt,
        'TP_px': int(tp_px),
        'FP_px': int(fp_px),
        'FN_px': int(fn_px),
    })





metrics_path = os.path.join(output_dir, "TP_FP_FN_pixelwise.xlsx")
metrics = pd.DataFrame(metrics_list)

# Store in excel
metrics.to_excel(metrics_path)

print(f"Saved Metrics excel to {metrics_path}")

