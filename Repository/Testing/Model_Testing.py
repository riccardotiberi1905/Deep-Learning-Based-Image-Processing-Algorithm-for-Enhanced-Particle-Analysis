import h5py

from MSR_NET import MSRNet, count_parameters, MSRNet_2_classes, MSRNet_TL
from Datasets_Testing import Dataset_Leyre, Dataset_Ricc, Dataset_Ricc_2_classes, Dataset_common
import torch
import os
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
from Help_Functions import compute_dice, compute_iou, extract_ordered_h5_testing, recompone
import pandas as pd

if __name__ == '__main__':

    '''
    if __name__ == '__main__': Guard:
    This condition ensures that the code inside it runs only when the script 
    is executed directly, not when it’s imported by another process. 
    This prevents unwanted re-execution of the main training code in the child processes.
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 128
    resume = True
    checkpoint_path = os.path.join('./checkpoint_ricc_def_cedice_FOV_small', 'net.pt')

    # ---------------------- Model Setup ------------------------
    net = MSRNet_2_classes()
    print("Total number of parameters: "+str(count_parameters(net)))
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.exists(checkpoint_path), "Checkpoint file not found!"
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']

    # ---------------------- Data Setup -----------------------
    test_dataset = Dataset_Ricc_2_classes()
    print("Total Data:", test_dataset.__len__())

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16)

    #criterion = torch.nn.CrossEntropyLoss()

    print("Starting predictions...")
    net.eval()
    predictions = []
    test_loss = 0.0


    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(test_loader):  # Ensure correct unpacking
            #inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.to(device)
            #targets = torch.squeeze(targets)
            outputs = net(inputs)
            #loss = criterion(outputs, targets)
            #test_loss += loss.item()
            #outputs = torch.nn.functional.softmax(outputs, dim=1)[:, 1, :, :]  # just for binary segmentation
            #outputs = (outputs > 0.5).float()  # Convert to binary mask (0 or 1)
            preds = torch.argmax(outputs, dim=1)  # Shape: [B, H, W]
            predictions.append(preds.cpu().numpy())

            #outputs = outputs.unsqueeze(1)
            #predictions.append(outputs.cpu().numpy())
    '''
    USE ONLY FOR TLOSS
    with torch.no_grad():
        for inputs, _ in test_loader:  # we ignore targets here
            inputs = inputs.to(device)
            logits = net(inputs)  # (B, H, W)
            probs = torch.sigmoid(logits)  # (B, H, W)
            preds = (probs > 0.5).cpu().numpy()  # numpy array [0,1]
            predictions.append(preds)
    '''

    prediction_patches = np.concatenate(predictions, axis=0)
    #print(test_loss)
    print("Predictions finished")
    print("Predictions Shape:", prediction_patches.shape)

    # ---------------------- Reconstruct patches into original images ------------------------
    # Paths to your saved HDF5 files
    input_h5_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\2_classes_ce_dice_FOV_small\test_input.h5"
    target_h5_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\2_classes_ce_dice_FOV_small\test_target.h5"

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
    base_dir = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\2_classes_ce_dice_FOV_small"
    save_target_path = os.path.join(base_dir, 'recon_target_def.h5')
    save_predict_path = os.path.join(base_dir, 'recon_predicted_def.h5')

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

    '''
    # ------------------------ Evaluation metrics ----------------------------
    results = []

    # Compute IoU and Dice scores for each image in the dataset
    print("Computing IoU and Dice scores...")
    for idx in range(len(reconstructed_imgs)):
        iou_scores = compute_iou(reconstructed_predictions[idx], reconstructed_masks[idx],
                                 num_classes=2)
        dice_scores = compute_dice(reconstructed_predictions[idx], reconstructed_masks[idx],
                                   num_classes=2)

        # Store the results
        results.append({
            "Image_ID": idx,  # Index of the image
            "IoU_White": iou_scores.get(1, float('nan')),  # IoU for white (foreground)
            "IoU_Black": iou_scores.get(0, float('nan')),  # IoU for black (background)
            "Dice_White": dice_scores.get(1, float('nan')),  # Dice for white (foreground)
            "Dice_Black": dice_scores.get(0, float('nan'))  # Dice for black (background)
        })

    # Convert to a DataFrame
    df_results = pd.DataFrame(results)

    # Save the results to a CSV file for analysis
    df_results.to_csv("iou_dice_scores.csv", index=False)

    # Print summary statistics
    print(df_results.describe())

    
    import matplotlib.pyplot as plt

    
    def visualize_predictions(reconstructed_imgs, reconstructed_masks, reconstructed_predictions, num_samples=10):

        # Squeeze the patches to remove the channel dimension
        img_patches = np.squeeze(reconstructed_imgs[:num_samples], axis=1)  # shape: (num_samples, 64, 64)
        mask_patches = np.squeeze(reconstructed_masks[:num_samples], axis=1)  # shape: (num_samples, 64, 64)
        prediction_patches = np.squeeze(reconstructed_predictions[:num_samples], axis=1)  # shape: (num_samples, 64, 64)

        # Set up the figure for visualization
        fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 2))

        for i in range(num_samples):
            # Plot original image
            axes[i, 0].imshow(img_patches[i], cmap='gray')
            axes[i, 0].set_title(f"Original Image {i + 1}")
            axes[i, 0].axis('off')

            # Plot mask
            axes[i, 1].imshow(mask_patches[i], cmap='gray')
            axes[i, 1].set_title(f"Mask {i + 1}")
            axes[i, 1].axis('off')

            # Plot prediction
            axes[i, 2].imshow(prediction_patches[i], cmap='gray')
            axes[i, 2].set_title(f"Prediction {i + 1}")
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.show()


    # Example usage:
    visualize_predictions(reconstructed_imgs, reconstructed_masks, reconstructed_predictions, num_samples=10)
    '''