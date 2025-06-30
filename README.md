# Deep-Learning-Based-Image-Processing-Algorithm-for-Enhanced-Particle-Analysis

Project Structure

Pre‑Processing and Data Preparation/

Pre_Processing.py
Resamples and normalizes your raw images and masks.

Prepare_Data_H5_Training.py & Prepare_Data_H5_Testing.py
Each script builds its own .h5 file, applying your chosen augmentations and splitting into train/test sets.

Training/

Datasets_Training.py
Wraps your .h5 data in a PyTorch‐compatible Dataset class.

Model_Training.py
Orchestrates the training loop—loading data, stepping through MSR_Net, computing loss (BCE, Dice, Focal, etc.), and checkpointing.

Models/
Your collection of loss functions, ready to import in any training script:

bce_loss.py

focal_loss.py

bce_dice_loss.py

Residual U‑Net/

MSR_Net.py
The core Residual U‑Net architecture you’re training and evaluating.

Testing/

Datasets_Testing.py
Mirrors the training dataset class but for your held‑out images.

Model_Testing.py
Loads a trained checkpoint, runs inference over the test set, and saves raw predictions.

Post_Processing_MOD.py
Cleans up the binary masks with morphological operations (e.g., closing, opening).

evaluation/
All your metric calculators and reporting tools—Dice, IoU, precision/recall, confusion matrices, size‐bin analyses, and plotting utilities.
