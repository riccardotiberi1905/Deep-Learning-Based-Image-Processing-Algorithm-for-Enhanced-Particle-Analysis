# Deep-Learning-Based-Image-Processing-Algorithm-for-Enhanced-Particle-Analysis

├── Pre-Processing and Data Preparation/ # Data loading, cleaning, augmentation, .h5 preparation
│ ├── Pre_Processing.py # Image Resampling
│ └── Prepare_Data_H5_Testing.py # Builds train/test .h5 datasets
│ └── Prepare_Data_H5_Training.py # Builds train/test .h5 datasets
│
├── Training # Training loop and dataset creation 
│ ├── Datasets_Training.py # Dataset
│ └── Model_Training.py # Training loop
│
├── Models # Models with weights
│ ├── focal_loss.py
│ ├── bce_loss.py
│ └── bce_dice_loss.py
│
├── Residual U-Net/ # Model architectures
│ └── MSR_Net.py # Residual U‑Net definition
│
├── Testing/ # Inference & post‑processing
│ └── Datasets_Testing.py # Dataset
│ └── Model_Testing.py # Testing loop
│ └── Post_Processing_MOD.py # morphological cleanup
│
├── evaluation/ # Metrics & reporting
