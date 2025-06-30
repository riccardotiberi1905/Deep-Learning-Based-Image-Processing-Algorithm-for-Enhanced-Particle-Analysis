import h5py
import os
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops

# Input paths
images_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Focal_Loss\test_1\post_process_predicted_v3.h5"
mask_path   = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Focal_Loss\test_1\post_process_target_v3.h5"
output_dir  = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Focal_Loss\test_1\Results_v3"

# Parameters
median_px_mm        = 29.98       # Pixel to mm conversion
distance_thresh     = 10          # Pixel distance for matching

# List to collect per-image metrics
metrics_list = []

# Load ordered patches
def load_ordered_patches(path):
    with h5py.File(path, 'r') as f:
        keys = sorted(f.keys(), key=lambda x: int(x))
        return [f[k][...] for k in keys]

pred_patches = load_ordered_patches(images_path)
gt_patches   = load_ordered_patches(mask_path)

# Iterate over each patch
for idx, (pred, gt) in enumerate(zip(pred_patches, gt_patches)):
    # Counters for this image
    gt_count = 0
    pred_count = 0
    match_count = 0

    # Label connected components
    pred_labels  = label(pred, connectivity=1)
    gt_labels    = label(gt, connectivity=1)
    pred_regions = regionprops(pred_labels)
    gt_regions   = regionprops(gt_labels)

    # Collect centroids for GT <1mm
    gt_centroids = []
    for region in gt_regions:
        length_mm = region.major_axis_length / median_px_mm
        if length_mm < 1:
            gt_count += 1
            gt_centroids.append(region.centroid)

    # Collect centroids for Pred <1mm
    pred_centroids = []
    for region in pred_regions:
        length_mm = region.major_axis_length / median_px_mm
        if length_mm < 1:
            pred_count += 1
            pred_centroids.append(region.centroid)

    # Match GT and Pred centroids within threshold
    matched_pred_indices = set()
    for gt_centroid in gt_centroids:
        # Compute distances to predicted centroids and track indices
        dists = [np.linalg.norm(np.array(gt_centroid) - np.array(pc)) for pc in pred_centroids]
        # Find the predicted match
        for i, d in enumerate(dists):
            if d < distance_thresh and i not in matched_pred_indices:
                match_count += 1
                matched_pred_indices.add(i)
                break

    # Calculate false positives and false negatives
    false_positives = pred_count - len(matched_pred_indices)
    false_negatives = gt_count - match_count

    # Append metrics for this image
    metrics_list.append({
        'Index': idx,
        'GT_emboli_lt1mm': gt_count,
        'Pred_emboli_lt1mm': pred_count,
        'Matched_emboli_lt1mm': match_count,
        'False_Positives': false_positives,
        'False_Negatives': false_negatives
    })

# Create DataFrame and save
df = pd.DataFrame(metrics_list)

os.makedirs(output_dir, exist_ok=True)
results_path = os.path.join(output_dir, "Emboli_Counts_PerImage_lt1mm.xlsx")
df.to_excel(results_path, index=False)
print(f"Per-image results with FP/FN saved to: {results_path}")

