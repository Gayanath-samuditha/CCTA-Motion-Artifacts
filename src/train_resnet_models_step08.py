"""
Train ResNet on 128x128 pre-corrupted patches, 0-10mm severity, 4 classes.
Patient-level 70/15/15 split. Flat patch directory.
Usage:
python train_offline_128_10mm.py --mode classification --backbone resnet34 --epochs 50 --num-workers 0
python train_offline_128_10mm.py --mode regression --backbone resnet50 --epochs 50 --num-workers 0
"""

import os
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, resnet34, resnet50
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
import SimpleITK as sitk

PATCHES_DIR = "/wrk-vakka/users/mohogaya/caisa/data/corrupted_patches_128_10mm"
OUTPUT_BASE = "/wrk-vakka/users/mohogaya/caisa/results/training_offline_128_10mm"


class OfflineDataset128(Dataset):
    #load 128x128 patches from flat directory
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
                self.data.append((os.path.join(patches_dir, fname), severity, severity_class))
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


class ResNetRegressor(nn.Module):
    def __init__(self, backbone='resnet18'):
        super().__init__()
        models = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50}
        self.backbone = models[backbone](pretrained=False)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.backbone(x).squeeze(-1)


class ResNetClassifier(nn.Module):
    def __init__(self, backbone='resnet18', num_classes=4):
        super().__init__()
        models = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50}
        self.backbone = models[backbone](pretrained=False)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def train_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return running_loss / len(loader)


def validate(model, loader, criterion, device, mode, epoch, total_epochs):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Val]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            if mode == 'regression':
                preds = outputs.cpu().numpy()
            else:
                preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = running_loss / len(loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if mode == 'regression':
        mae = np.mean(np.abs(all_preds - all_labels))
        rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
        return avg_loss, {'MAE': mae, 'RMSE': rmse}
    else:
        accuracy = np.mean(all_preds == all_labels)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        cm = confusion_matrix(all_labels, all_preds)
        return avg_loss, {'Accuracy': accuracy, 'F1_Score': f1, 'Confusion_Matrix': cm}


def plot_confusion_matrix(cm, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    classes = ['Clean', 'Mild', 'Moderate', 'Severe']
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix', ylabel='True', xlabel='Predicted')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_training(train_losses, val_losses, val_metrics, mode, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    if mode == 'regression':
        mae = [m['MAE'] for m in val_metrics]
        axes[1].plot(mae, color='orange')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Validation MAE')
    else:
        acc = [m['Accuracy'] for m in val_metrics]
        f1 = [m['F1_Score'] for m in val_metrics]
        axes[1].plot(acc, label='Accuracy')
        axes[1].plot(f1, label='F1 Score')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Validation Metrics')
        axes[1].legend()
    axes[1].set_xlabel('Epoch')
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patches-dir', type=str, default=PATCHES_DIR)
    parser.add_argument('--output-dir', type=str, default=OUTPUT_BASE)
    parser.add_argument('--mode', type=str, default='classification',
                        choices=['regression', 'classification', 'binary'])
    parser.add_argument('--backbone', type=str, default='resnet34',
                        choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"{args.mode}_{args.backbone}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    #patient-level 70/15/15 split from flat directory
    all_case_ids = set()
    for fname in os.listdir(args.patches_dir):
        if fname.endswith('.nii.gz'):
            try:
                all_case_ids.add(int(fname.split('_')[1]))
            except:
                continue

    all_cases = np.array(sorted(all_case_ids))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(all_cases)
    n = len(all_cases)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train_cases = sorted(all_cases[:n_train].tolist())
    val_cases = sorted(all_cases[n_train:n_train + n_val].tolist())
    test_cases = sorted(all_cases[n_train + n_val:].tolist())
    print(f"Patient split — Train: {len(train_cases)}, Val: {len(val_cases)}, Test: {len(test_cases)}")

    #save splits
    with open(output_dir / 'data_splits.json', 'w') as f:
        json.dump({'train': train_cases, 'val': val_cases, 'test': test_cases}, f)

    train_dataset = OfflineDataset128(args.patches_dir, train_cases, args.mode)
    val_dataset = OfflineDataset128(args.patches_dir, val_cases, args.mode)
    test_dataset = OfflineDataset128(args.patches_dir, test_cases, args.mode)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    print(f"Patches — Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    if args.mode == 'regression':
        model = ResNetRegressor(args.backbone)
        criterion = nn.SmoothL1Loss()
    else:
        num_classes = 2 if args.mode == 'binary' else 4
        model = ResNetClassifier(args.backbone, num_classes)
        label_smooth = 0.1 if num_classes > 2 else 0.0
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_losses, val_losses, val_metrics = [], [], []
    best_val_loss = float('inf')
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        train_losses.append(train_loss)

        val_loss, metrics = validate(model, val_loader, criterion, device, args.mode, epoch, args.epochs)
        val_losses.append(val_loss)
        val_metrics.append(metrics)
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{args.epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}, LR={lr:.6f}")
        for k, v in metrics.items():
            if k != 'Confusion_Matrix':
                print(f"  {k}: {v:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            print(f"  → Saved best")

        if epoch % 5 == 0:
            plot_training(train_losses, val_losses, val_metrics, args.mode,
                          output_dir / 'training_curves.png')
            if args.mode != 'regression':
                plot_confusion_matrix(metrics['Confusion_Matrix'],
                                      output_dir / f'cm_epoch{epoch}.png')

    print(f"\nTraining done: {(time.time()-start_time)/60:.1f} min, Best epoch: {best_epoch}")
    plot_training(train_losses, val_losses, val_metrics, args.mode, output_dir / 'training_curves.png')
    if args.mode != 'regression':
        plot_confusion_matrix(metrics['Confusion_Matrix'], output_dir / 'cm_final.png')
    torch.save(model.state_dict(), output_dir / 'final_model.pth')
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
