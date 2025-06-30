import h5py
import os
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
import imageio

#------------------------- Essential Functions -------------------------------
def extract_ordered_h5_testing(images_path, mask_path):
    """
    Extract ordered image and mask patches from h5 files for further reconstruction
    """
    with h5py.File(images_path, 'r') as f:
        keys = sorted(f.keys(), key=lambda x: int(x))
        image_patches = np.stack([f[k][...] for k in keys], axis=0)
        print("Prediction Images Shape:", image_patches.shape)

    with h5py.File(mask_path, 'r') as f:
        keys = sorted(f.keys(), key=lambda x: int(x))
        mask_patches = np.stack([f[k][...] for k in keys], axis=0)
        print("Target Masks Shape:", mask_patches.shape)

    return image_patches, mask_patches

#---------------------- Define Directories -----------------------------------
prediction_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\1_class_tloss_FOV_small\test\post_process_predicted.h5"
mask_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\1_class_tloss_FOV_small\test\post_process_target.h5"
output_dir = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\1_class_tloss_FOV_small\test\Results\Threshold_5pixels"
metrics_path    = os.path.join(output_dir, "FP_FN.xlsx")
fp_images_dir   = os.path.join(output_dir, "FPN_images")

# Define size bins in microns (edges) and labels
size_bins = [70, 100, 200, 300, 400, 500, 600, 700, 800, 900,
             1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
bin_labels = [f"{size_bins[i]}-{size_bins[i+1]}" if i < len(size_bins)-1 else f">={size_bins[i]}"
              for i in range(len(size_bins))]

median_px_mm = 29.98  # Pixel to mm conversion
y_threshold = 5     # Pixel distance threshold for matching

# Load arrays
os.makedirs(fp_images_dir, exist_ok=True)
preds, gts = extract_ordered_h5_testing(prediction_path, mask_path)

metrics_list = []

for idx, (pred_arr, gt_arr) in enumerate(zip(preds, gts)):
    # Initialize counters
    counts = {f'FP_{lbl}': 0 for lbl in bin_labels}
    counts.update({f'FN_{lbl}': 0 for lbl in bin_labels})

    # Label particles
    label_pred = label(pred_arr, connectivity=1)
    label_gt   = label(gt_arr, connectivity=1)
    regions_pred = regionprops(label_pred)
    regions_gt   = regionprops(label_gt)

    cent_pred = np.array([r.centroid for r in regions_pred])
    cent_gt   = np.array([r.centroid for r in regions_gt])

    # Track regions
    large_flag = False
    all_fp = []
    all_fn = []

    # GT -> Pred (False Negatives)
    for gt in regions_gt:
        dists = np.linalg.norm(cent_pred - gt.centroid, axis=1) if cent_pred.size else np.array([])
        match_idx = np.argmin(dists) if dists.size else None
        matched = dists[match_idx] < y_threshold if dists.size else False
        size_micron = gt.major_axis_length / median_px_mm * 1000
        if not matched:
            all_fn.append(gt.label)
            bin_idx = np.digitize(size_micron, size_bins) - 1
            lbl = bin_labels[min(bin_idx, len(bin_labels)-1)]
            counts[f'FN_{lbl}'] += 1
            if size_micron >= 1000:
                large_flag = True

    # Pred -> GT (False Positives)
    for pr in regions_pred:
        dists = np.linalg.norm(cent_gt - pr.centroid, axis=1) if cent_gt.size else np.array([])
        match_idx = np.argmin(dists) if dists.size else None
        matched = dists[match_idx] < y_threshold if dists.size else False
        size_micron = pr.major_axis_length / median_px_mm * 1000
        if not matched:
            all_fp.append(pr.label)
            bin_idx = np.digitize(size_micron, size_bins) - 1
            lbl = bin_labels[min(bin_idx, len(bin_labels)-1)]
            counts[f'FP_{lbl}'] += 1
            if size_micron >= 1000:
                large_flag = True

    # Save overlay image if any FP/FN >= 1000
    if large_flag:
        h, w = pred_arr.shape[-2:]
        # Blank canvas
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        # Predicted: white translucent
        mask_pred = label_pred > 0
        alpha_pred = 0.3
        rgb[mask_pred] = rgb[mask_pred] * (1 - alpha_pred) + np.array([1., 1., 1.]) * alpha_pred
        # Ground truth: green translucent
        mask_gt = label_gt > 0
        alpha_gt = 0.3
        rgb[mask_gt] = rgb[mask_gt] * (1 - alpha_gt) + np.array([0., 1., 0.]) * alpha_gt

        # False positives: red translucent
        alpha_fp = 0.5
        for lab in all_fp:
            mask = label_pred == lab
            rgb[mask] = rgb[mask] * (1 - alpha_fp) + np.array([1., 0., 0.]) * alpha_fp
        # False negatives: blue translucent
        alpha_fn = 0.5
        for lab in all_fn:
            mask = label_gt == lab
            rgb[mask] = rgb[mask] * (1 - alpha_fn) + np.array([0., 0., 1.]) * alpha_fn

        # Convert to uint8 and save
        rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        img_path = os.path.join(fp_images_dir, f"idx_{idx}_FPN_overlay.png")
        imageio.imwrite(img_path, rgb_uint8)

    # Append metrics
    metrics_list.append({
        'Index': idx,
        'Total_Pred': len(regions_pred),
        'Total_GT': len(regions_gt),
        **counts
    })

# Save metrics to Excel
os.makedirs(output_dir, exist_ok=True)
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_excel(metrics_path, index=False)
print(f"Saved Metrics to {metrics_path}")
