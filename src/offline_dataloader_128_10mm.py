"""
Dataloader for 128x128 corrupted patches, 0-10mm severity, 4 classes.
Patient-level 70/15/15 split. No center crop - uses full 128x128.
Flat directory: case_0001_patch_000_sev0.00_class0.nii.gz
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk


class OfflineDataset128(Dataset):
    #Load 128x128 pre-corrupted patches from flat directory
    def __init__(self, patches_dir, case_ids, mode):
        self.mode = mode
        self.data = []

        case_id_set = set(case_ids)

        for fname in sorted(os.listdir(patches_dir)):
            if not fname.endswith('.nii.gz'):
                continue
            try:
                #parse: case_0001_patch_000_sev0.00_class0.nii.gz
                parts = fname.replace('.nii.gz', '').split('_')
                case_id = int(parts[1])
                if case_id not in case_id_set:
                    continue
                severity = float(parts[4].replace('sev', ''))
                severity_class = int(parts[5].replace('class', ''))
                path = os.path.join(patches_dir, fname)
                self.data.append((path, severity, severity_class))
            except:
                continue

        print(f"Loaded {len(self.data)} patches from {len(case_ids)} cases")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, severity, severity_class = self.data[idx]

        img = sitk.ReadImage(path)
        patch = sitk.GetArrayFromImage(img)[0]  #(128, 128)

        #normalize
        pmin, pmax = patch.min(), patch.max()
        if pmax > pmin:
            patch = (patch - pmin) / (pmax - pmin)

        image = torch.from_numpy(patch).unsqueeze(0).float()  #(1, 128, 128)

        if self.mode == 'regression':
            label = torch.tensor(severity, dtype=torch.float32)
        elif self.mode == 'classification':
            label = torch.tensor(severity_class, dtype=torch.long)
        elif self.mode == 'binary':
            label = torch.tensor(1 if severity > 0 else 0, dtype=torch.long)

        return image, label


def create_dataloaders(patches_dir, mode, batch_size, num_workers, seed,
                       train_ratio=0.70, val_ratio=0.15):
    #patient-level 70/15/15 split
    np.random.seed(seed)

    #extract unique case IDs from filenames
    all_case_ids = set()
    for fname in os.listdir(patches_dir):
        if not fname.endswith('.nii.gz'):
            continue
        try:
            case_id = int(fname.split('_')[1])
            all_case_ids.add(case_id)
        except:
            continue

    all_cases = np.array(sorted(all_case_ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(all_cases)

    n = len(all_cases)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_cases = sorted(all_cases[:n_train].tolist())
    val_cases = sorted(all_cases[n_train:n_train + n_val].tolist())
    test_cases = sorted(all_cases[n_train + n_val:].tolist())

    print(f"Patient split — Train: {len(train_cases)}, Val: {len(val_cases)}, Test: {len(test_cases)}")

    train_dataset = OfflineDataset128(patches_dir, train_cases, mode)
    val_dataset = OfflineDataset128(patches_dir, val_cases, mode)
    test_dataset = OfflineDataset128(patches_dir, test_cases, mode)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"Patches — Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
