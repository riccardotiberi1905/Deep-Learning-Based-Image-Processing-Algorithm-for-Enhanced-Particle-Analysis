import h5py
import torch
import os
from torch.utils.data import DataLoader
from MSR_NET import MSRNet_2_classes
from Datasets_Testing import Dataset_Ricc_2_classes
from Help_Functions import extract_ordered_h5_testing, recompone
import numpy as np

# Assume you already have:
#   - three model architectures instantiated identically (e.g. model_CE, model_CE_Dice, model_Focal)
#   - their checkpoints loaded: model_CE.load_state_dict(...), etc.
#   - a validation/test Dataset + DataLoader that yields (image_tensor, label_tensor) if you want to measure performance.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = []
checkpoint_path1 = os.path.join('./checkpoint_ricc_def_cedice_FOV_small', 'net.pt')
checkpoint_path2 = os.path.join('./checkpoint_ricc_focal_loss_v2', 'net.pt')
checkpoint_path3 = os.path.join('./checkpoint_ricc_def', 'net.pt')
for ckpt_path in [checkpoint_path1, checkpoint_path2, checkpoint_path3]:
    # 1) load the entire checkpoint dict
    checkpoint = torch.load(ckpt_path, map_location=device)
    # 2) instantiate your network
    net = MSRNet_2_classes()
    # 3) load only the 'net' sub‐dict
    net.load_state_dict(checkpoint['net'])
    net.eval()
    net.to(device)
    models.append(net)

# If you want to weight them unequally, define a list of weights (must sum to 1.0).
# For equal weighting:
weights = [1/3, 1/3, 1/3]
# Or—for example—if the CE+Dice model is slightly stronger:
# weights = [0.25, 0.5, 0.25]  # CE, CE+Dice, Focal

def ensemble_predict(input_tensor):
    """
    Given a batch of images (B×C×H×W), run all three models and return
    the averaged probability map (B×num_classes×H×W).
    """
    # Move to device
    x = input_tensor.to(device)
    # Collect softmax outputs
    prob_maps = []
    with torch.no_grad():
        for m in models:
            logits = m(x)            # shape: (B, num_classes, H, W)
            probs = torch.softmax(logits, dim=1)
            prob_maps.append(probs)
    # Weighted average of the probability maps
    # Stack: shape (3, B, num_classes, H, W) if you do prob_maps = torch.stack(prob_maps)
    stacked = torch.stack(prob_maps, dim=0)  # (3, B, C, H, W)
    w_tensor = torch.tensor(weights, device=device).view(3, 1, 1, 1, 1)
    # Multiply each model's prob map by its weight and sum:
    P_ens = (w_tensor * stacked).sum(dim=0)   # shape: (B, C, H, W)
    return P_ens  # You can later do argmax over dim=1 to get hard mask.

if __name__ == '__main__':
    # Example: iterate over a DataLoader to save ensembled outputs or compute metrics
    batch_size = 128
    test_dataset = Dataset_Ricc_2_classes()
    print("Total Data:", test_dataset.__len__())
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16)

    print("Starting predictions...")
    net.eval()
    predictions = []
    test_loss = 0.0


    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(test_loader):  # Ensure correct unpacking
            inputs = inputs.to(device)
            # 2) Run all three models and get the ensembled probability map
            P_ens = ensemble_predict(inputs)  # shape: (B, num_classes, H, W)

            # 3) Convert the averaged softmax into a hard mask
            preds = torch.argmax(P_ens, dim=1)  # shape: (B, H, W)

            # 4) “preds” is now a NumPy array of size (batch_size, H, W) once we .cpu()
            predictions.append(preds.cpu().numpy())

    prediction_patches = np.concatenate(predictions, axis=0)
    print("Predictions finished")
    print("Predictions Shape:", prediction_patches.shape)

    # ---------------------- Reconstruct patches into original images ------------------------
    # Paths to your saved HDF5 files
    input_h5_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Ensemble_v2\test_input.h5"
    target_h5_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Ensemble_v2\test_target.h5"

    # Load input and target data
    test_image_patches, test_mask_patches = extract_ordered_h5_testing(input_h5_path, target_h5_path)

    # Reconstruct patches
    print("Reconstructing images...")
    reconstructed_imgs, reconstructed_masks, reconstructed_predictions = recompone(
            test_image_patches,
            test_mask_patches,
            prediction_patches,
            full_img_shape=(896, 896),
            patch_size=(64, 64)
        )

    # Normalize reconstructed masks
    reconstructed_masks = reconstructed_masks/255.

    # Store predicted images and reconstructed masks
    # Define Base Directory to store H5 files
    base_dir = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Ensemble_v2"
    save_target_path = os.path.join(base_dir, 'recon_target_ensemble_v2.h5')
    save_predict_path = os.path.join(base_dir, 'recon_predicted_ensemble_v2.h5')

    target = reconstructed_masks
    predictions = reconstructed_predictions
    print(target.shape)
    print(predictions.shape)

    target_h5f = h5py.File(save_target_path, 'w')
    predicted_h5f = h5py.File(save_predict_path, 'w')

    total_number = len(target)
    for i in range(total_number):
            # print(i)
            target_data = target[i, ...]
            target_h5f.create_dataset(str(i), data=target_data)

            predicted_data = predictions[i, ...]
            predicted_h5f.create_dataset(str(i), data=predicted_data)
    target_h5f.close()
    predicted_h5f.close()
