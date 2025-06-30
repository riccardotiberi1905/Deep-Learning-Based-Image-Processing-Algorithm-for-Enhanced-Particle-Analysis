import h5py
import torch
import os
from torch.utils.data import DataLoader
from MSR_NET import MSRNet_2_classes
from Datasets_Testing import Dataset_Ricc_2_classes
from Help_Functions import extract_ordered_h5_testing, recompone
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoints and build models list
checkpoint_paths = [
    os.path.join('./checkpoint_ricc_def_cedice_FOV_small', 'net.pt'),
    os.path.join('./checkpoint_ricc_focal_loss_v2', 'net.pt'),
    os.path.join('./checkpoint_ricc_def', 'net.pt')
]
models = []
for ckpt_path in checkpoint_paths:
    checkpoint = torch.load(ckpt_path, map_location=device)
    net = MSRNet_2_classes()
    net.load_state_dict(checkpoint['net'])
    net.eval()
    net.to(device)
    models.append(net)

# Majority voting function
def ensemble_majority(input_tensor):
    """
    Given a batch of images (B×C×H×W), run all three models and return
    the majority-voted hard mask (B×H×W).
    """
    x = input_tensor.to(device)
    hard_preds = []
    with torch.no_grad():
        for m in models:
            logits = m(x)                     # (B, num_classes, H, W)
            preds = torch.argmax(logits, dim=1)  # (B, H, W)
            hard_preds.append(preds)
    # Stack predictions: (3, B, H, W) -> (B, 3, H, W)
    stacked = torch.stack(hard_preds, dim=1)  # shape: (B, 3, H, W)
    # Compute majority vote along dim=1
    majority, _ = torch.mode(stacked, dim=1)  # (B, H, W)
    return majority

if __name__ == '__main__':
    # Create DataLoader
    batch_size = 128
    test_dataset = Dataset_Ricc_2_classes()
    print("Total Data:", len(test_dataset))
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=16
    )

    # Run inference with majority voting
    print("Starting majority-vote predictions...")
    all_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            preds = ensemble_majority(inputs)     # (B, H, W)
            all_preds.append(preds.cpu().numpy())

    prediction_patches = np.concatenate(all_preds, axis=0)
    print("Predictions finished")
    print("Predictions Shape:", prediction_patches.shape)

    # Reconstruct patches into full images
    input_h5_path  = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Ensemble_v4\test_input.h5"
    target_h5_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Ensemble_v4\test_target.h5"
    test_image_patches, test_mask_patches = extract_ordered_h5_testing(input_h5_path, target_h5_path)

    print("Reconstructing images...")
    imgs, masks, preds_full = recompone(
        test_image_patches,
        test_mask_patches,
        prediction_patches,
        full_img_shape=(896, 896),
        patch_size=(64, 64)
    )

    # Normalize masks\m
    reconstructed_masks = masks / 255.0

    # Save to HDF5
    base_dir = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Ensemble_v4"
    save_target = os.path.join(base_dir, 'recon_target_ensemble_v4.h5')
    save_pred   = os.path.join(base_dir, 'recon_predicted_ensemble_v4.h5')

    # Write datasets
    with h5py.File(save_target, 'w') as f_tar, h5py.File(save_pred, 'w') as f_pred:
        for i, (t, p) in enumerate(zip(reconstructed_masks, preds_full)):
            f_tar.create_dataset(str(i), data=t)
            f_pred.create_dataset(str(i), data=p)

    print("Saved majority-vote H5 files.")
