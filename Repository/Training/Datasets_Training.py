import h5py
import os
import numpy as np
import torch
import random
import torch.utils.data as udata

class Dataset_Leyre(udata.Dataset):
    def __init__(self):
        super(Dataset_Leyre, self).__init__()

        data_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Leyre"

        self.target_path = os.path.join(data_path, 'train_target.h5')
        self.input_path = os.path.join(data_path, 'train_input.h5')

        target_h5f = h5py.File(self.target_path, 'r')
        input_h5f = h5py.File(self.input_path, 'r')

        input_keys = list(input_h5f.keys())
        target_keys = list(target_h5f.keys())

        print("Number of input patches:", len(input_keys))
        print("Number of target patches:", len(target_keys))

        assert len(input_keys) == len(target_keys), "Mismatch in number of patches!"

        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)

        target_h5f.close()
        input_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        target_h5f = h5py.File(self.target_path, 'r')
        input_h5f = h5py.File(self.input_path, 'r')

        key = self.keys[index]
        target = np.array(target_h5f[key])
        target[target < 128] = [0]
        target[target >= 128] = [255]
        target = torch.from_numpy(target/255.).long()
        input = np.array(input_h5f[key])
        input = torch.from_numpy(input/255.).float()
        target_h5f.close()
        input_h5f.close()

        return input, target

class Dataset_Ricc_1_class(udata.Dataset):
    def __init__(self):
        super(Dataset_Ricc_1_class, self).__init__()

        data_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\1_class_tloss_FOV_small"

        self.target_path = os.path.join(data_path, 'train_target.h5')
        self.input_path = os.path.join(data_path, 'train_input.h5')

        target_h5f = h5py.File(self.target_path, 'r')
        input_h5f = h5py.File(self.input_path, 'r')

        input_keys = list(input_h5f.keys())
        target_keys = list(target_h5f.keys())

        print("Number of input patches:", len(input_keys))
        print("Number of target patches:", len(target_keys))

        assert len(input_keys) == len(target_keys), "Mismatch in number of patches!"

        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)

        target_h5f.close()
        input_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        target_h5f = h5py.File(self.target_path, 'r')
        input_h5f = h5py.File(self.input_path, 'r')

        key = self.keys[index]
        target = np.array(target_h5f[key])
        target[target < 128] = [0]
        target[target >= 128] = [255]
        target = torch.from_numpy(target/255.).long()
        input = np.array(input_h5f[key])
        input = torch.from_numpy(input/255.).float()
        target_h5f.close()
        input_h5f.close()

        return input, target

class Dataset_Ricc_2_classes(udata.Dataset):
    def __init__(self):
        super(Dataset_Ricc_2_classes, self).__init__()

        data_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\Focal_Loss"

        self.target_path = os.path.join(data_path, 'train_target.h5')
        self.input_path = os.path.join(data_path, 'train_input.h5')

        target_h5f = h5py.File(self.target_path, 'r')
        input_h5f = h5py.File(self.input_path, 'r')

        input_keys = list(input_h5f.keys())
        target_keys = list(target_h5f.keys())

        print("Number of input patches:", len(input_keys))
        print("Number of target patches:", len(target_keys))

        assert len(input_keys) == len(target_keys), "Mismatch in number of patches!"

        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)

        target_h5f.close()
        input_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        target_h5f = h5py.File(self.target_path, 'r')
        input_h5f = h5py.File(self.input_path, 'r')

        key = self.keys[index]
        target = np.array(target_h5f[key])

        middle_class_range = (80, 100)
        mask_bin = np.zeros_like(target, dtype=np.uint8)
        mask_bin[(target >= middle_class_range[0]) & (target <= middle_class_range[1])] = 1  # Middle class
        mask_bin[target > 128] = 2  # High-intensity class

        # Convert the numpy array to a torch tensor of type long
        target = torch.from_numpy(mask_bin).long()

        input = np.array(input_h5f[key])
        input = torch.from_numpy(input/255.).float()
        target_h5f.close()
        input_h5f.close()

        return input, target


class Dataset_common(udata.Dataset):
    def __init__(self):
        super(Dataset_common, self).__init__()

        data_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\common_dataset"

        self.target_path = os.path.join(data_path, 'train_target.h5')
        self.input_path = os.path.join(data_path, 'train_input.h5')

        target_h5f = h5py.File(self.target_path, 'r')
        input_h5f = h5py.File(self.input_path, 'r')

        input_keys = list(input_h5f.keys())
        target_keys = list(target_h5f.keys())

        print("Number of input patches:", len(input_keys))
        print("Number of target patches:", len(target_keys))

        assert len(input_keys) == len(target_keys), "Mismatch in number of patches!"

        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)

        target_h5f.close()
        input_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        target_h5f = h5py.File(self.target_path, 'r')
        input_h5f = h5py.File(self.input_path, 'r')

        key = self.keys[index]
        target = np.array(target_h5f[key])

        middle_class_range = (80, 100)
        mask_bin = np.zeros_like(target, dtype=np.uint8)
        mask_bin[(target >= middle_class_range[0]) & (target <= middle_class_range[1])] = 1  # Middle class
        mask_bin[target > 128] = 2  # High-intensity class

        # Convert the numpy array to a torch tensor of type long
        target = torch.from_numpy(mask_bin).long()

        input = np.array(input_h5f[key])
        input = torch.from_numpy(input/255.).float()
        target_h5f.close()
        input_h5f.close()

        return input, target

