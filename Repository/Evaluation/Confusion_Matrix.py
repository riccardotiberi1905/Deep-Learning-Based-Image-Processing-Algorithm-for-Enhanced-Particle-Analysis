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
prediction_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Focal_Loss\test\post_process_predicted.h5"
mask_path  = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Focal_Loss\test\post_process_target.h5"
# Determine output directory (same as input's folder)
output_dir = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Focal_Loss\test\Results\Threshold_10pixels"
#---------------------- Run Script -----------------------------------

# Evaluate particle matching when both masks have particles
threshold = 10  # Pixel distance threshold for matching

#1. Read h5 files
new_predictions_array, new_ground_truths_array = extract_ordered_h5_testing(prediction_path, mask_path)

# Initialize an empty list to store individual rows of metrics
median_px_mm = 29.98 #Pixel -- mm relationship
metrics_list = []

# 1) Define the shared bin‐edges (mm)
edges = [0,   0.07, 0.1] + list(np.arange(0.2, 2.1, 0.1)) + [np.inf]
# Build human‐readable labels
labels = []
for lo, hi in zip(edges[:-1], edges[1:]):
    if hi == np.inf:
        labels.append(f'>{lo:.2f}')
    else:
        labels.append(f'{lo:.2f}-{hi:.2f}')

# 2) Create an empty counts table (GT bins × Pred bins)
conf_mat = pd.DataFrame(
    0,
    index=labels,
    columns=labels,
    dtype=int
)

for index in range(len(new_predictions_array)):

    # Label connected components
    label_pred = label(new_predictions_array[index], connectivity=1)
    particles_pred = regionprops(label_pred)
    total_pred = len(particles_pred)

    label_mask = label(new_ground_truths_array[index], connectivity=1)
    particles_mask = regionprops(label_mask)
    total_mask = len(particles_mask)

    # Handle edge cases when one of the masks has no particles

    if total_mask == 0 and total_pred > 0:
        continue
    if total_mask > 0 and total_pred == 0:
        continue
    if total_mask == 0 and total_pred == 0:
        continue

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
                length_gt = np.round((gt_region.major_axis_length / median_px_mm), 3)
                length_pred = np.round((best_match.major_axis_length / median_px_mm), 3)
                # 3) Find the bin‐indices (np.digitize returns 1‐based indices)
                gt_bin_idx = np.digitize(length_gt, edges) - 1
                pred_bin_idx = np.digitize(length_pred, edges) - 1

                # 4) Convert to label strings
                gt_label = labels[gt_bin_idx]
                pred_label = labels[pred_bin_idx]

                # 5) Increment the corresponding cell
                conf_mat.loc[gt_label, pred_label] += 1

conf_path = os.path.join(output_dir, "confusion_matrix.xlsx")
conf_mat.to_excel(conf_path)
print(f"Saved confusion matrix to {conf_path}")
print(conf_mat)

import matplotlib.pyplot as plt
# 2) Prepare index array and counts
n = len(conf_mat)
idx = np.arange(n)
counts = conf_mat.values.astype(int)

# 3) Build a “severity” array but mask out diagonal and zero‐count cells
#    severity = |i–j| / max(|i–j|), but only where count>0 and i!=j
severity = np.abs(idx[:, None] - idx[None, :]).astype(float)
severity /= severity.max()
mask = (counts > 0) & (severity > 0)  # only mismatches with at least one match
severity_masked = np.where(mask, severity, np.nan)

# 4) Plot
fig, ax = plt.subplots(figsize=(10, 8))

# draw only the masked severity values
cax = ax.imshow(severity_masked,
                interpolation='nearest',
                cmap='Reds',
                vmin=0, vmax=1)

# annotate every cell: show count if >0, else leave blank
for i in range(n):
    for j in range(n):
        cnt = counts[i, j]
        if cnt > 0:
            ax.text(j, i, str(cnt),
                    ha='center', va='center',
                    color='white' if mask[i, j] and severity[i, j] > 0.5 else 'black',
                    fontsize=8)

# tick labels
ax.set_xticks(idx)
ax.set_xticklabels(conf_mat.columns, rotation=90, fontsize=8)
ax.set_yticks(idx)
ax.set_yticklabels(conf_mat.index, fontsize=8)
ax.set_xlabel("Predicted Size Bin (mm)")
ax.set_ylabel("Ground-Truth Size Bin (mm)")

# colorbar (for the mismatch severity scale)
cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Bin-Mismatch Severity (|i–j| normalized)")

plt.tight_layout()

# 5) Save to PNG
png_path = os.path.join(output_dir, "confusion_matrix.png")
fig.savefig(png_path, dpi=300)
plt.close(fig)

print(f"Saved filtered mismatch matrix to {png_path}")

# 2) Prepare index array and counts
n = len(conf_mat)
idx = np.arange(n)
counts = conf_mat.values.astype(int)

# 3) Compute severity for off‐diagonal mismatches
severity = np.abs(idx[:, None] - idx[None, :]).astype(float)
severity /= severity.max()
mask_mismatch = (counts > 0) & (severity > 0)
severity_masked = np.where(mask_mismatch, severity, np.nan)

# 4) Diagonal mask for correct matches (only if count>0)
mask_diag = np.eye(n, dtype=bool) & (counts > 0)

# 5) Plot
fig, ax = plt.subplots(figsize=(10, 8))

# 5a) Draw diagonal green first (alpha for lightness)
diag_cmap = plt.cm.Greens
ax.imshow(mask_diag.astype(float),
          interpolation='nearest',
          cmap=diag_cmap,
          vmin=0, vmax=1,
          alpha=0.4)

# 5b) Overlay mismatches in Reds
red_cmap = plt.cm.Reds
cax = ax.imshow(severity_masked,
                interpolation='nearest',
                cmap=red_cmap,
                vmin=0, vmax=1)

# 6) Annotate each nonzero cell
for i in range(n):
    for j in range(n):
        cnt = counts[i, j]
        if cnt > 0:
            # choose text color for contrast
            if mask_mismatch[i, j] and severity[i, j] > 0.5:
                txt_color = 'white'
            else:
                txt_color = 'black'
            ax.text(j, i, str(cnt),
                    ha='center', va='center',
                    color=txt_color, fontsize=8)

# 7) Ticks & labels
ax.set_xticks(idx)
ax.set_xticklabels(conf_mat.columns, rotation=90, fontsize=8)
ax.set_yticks(idx)
ax.set_yticklabels(conf_mat.index, fontsize=8)
ax.set_xlabel("Predicted Size Bin (mm)")
ax.set_ylabel("Ground‐Truth Size Bin (mm)")

# 8) Colorbar for mismatch severity
cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Bin‐Mismatch Severity (|i–j| normalized)")

plt.tight_layout()

# 9) Save to PNG
png_path = os.path.join(output_dir, "confusion_matrix_GREEN.png")
fig.savefig(png_path, dpi=300)
plt.close(fig)

print(f"Saved annotated confusion matrix to {png_path}")

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

# --- after you build your confusion‐matrix and save it ---

# 1) First, re‐scan your h5 to get the sorted list of patch‐keys
with h5py.File(prediction_path, 'r') as f_pred:
    image_keys = sorted(f_pred.keys(), key=lambda x: int(x))

# 2) While you were matching, collect all indices with a “huge” mismatch
#    Let’s call huge = bin‐index difference ≥ 2
huge_mismatch_indices = set()
# (Re‐run the matching logic in minimal form:)
for idx in range(len(new_predictions_array)):
    # label + regionprops
    lp = label(new_predictions_array[idx], connectivity=1)
    lm = label(new_ground_truths_array[idx], connectivity=1)
    props_p = regionprops(lp)
    props_m = regionprops(lm)
    for i, gm in enumerate(props_m):
        # find best bidirectional match as you already do…
        # (copy your matching block, up to computing gt_bin_idx / pred_bin_idx)
        centroid_m = np.array(gm.centroid)
        dists = [np.linalg.norm(centroid_m - np.array(pp.centroid)) for pp in props_p]
        matches = [j for j,d in enumerate(dists) if d< threshold]
        if not matches:
            continue
        best_j = matches[np.argmin([dists[j] for j in matches])]
        pp = props_p[best_j]
        # bidirectional check
        centroid_p = np.array(pp.centroid)
        rd = [np.linalg.norm(centroid_p - np.array(om.centroid)) for om in props_m]
        rmatch = [j for j,d in enumerate(rd) if d< threshold]
        if rmatch and rmatch[np.argmin([rd[j] for j in rmatch])] == i:
            # get bin‐indices
            length_m = np.round(gm.major_axis_length/median_px_mm, 3)
            length_p = np.round(pp.major_axis_length/median_px_mm, 3)
            bim = np.digitize(length_m, edges)-1
            bip = np.digitize(length_p, edges)-1
            if abs(bim - bip) >= 5:
                huge_mismatch_indices.add(idx)
                break   # one is enough to flag

# 3) Now for each flagged patch, build and save the overlay
for idx in huge_mismatch_indices:
    key = image_keys[idx]
    # read that single patch
    with h5py.File(prediction_path, 'r') as f_pred, \
         h5py.File(mask_path, 'r') as f_mask:
        pred_mask = f_pred[key][...].astype(bool)
        gt_mask   = f_mask[key][...].astype(bool)

    # build an RGB canvas
    H, W = pred_mask.shape[-2:]
    canvas = np.zeros((H, W, 3), dtype=float)

    # overlay GT in green (G channel), alpha=0.5
    canvas[..., 1] = np.where(gt_mask, 0.5, canvas[...,1])

    # overlay Pred in white (all channels), alpha=0.5
    for c in range(3):
        canvas[..., c] = np.where(pred_mask,
                                 canvas[...,c] + 0.5,
                                 canvas[...,c])
    # clip max at 1
    canvas = np.clip(canvas, 0, 1)

    # save
    out_name = f"{key}_mismatch_overlap.png"
    output_dir_images = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Focal_Loss\test\Results\Threshold_10pixels\mismatch_images"
    out_path = os.path.join(output_dir_images, out_name)
    plt.imsave(out_path, canvas)
    print(f"Saved overlap image: {out_path}")
