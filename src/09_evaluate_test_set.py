
#09_test_with_scatter.py
#Re-runs test evaluation and additionally saves a predicted-vs-true scatter plot for regression
#Classification path unchanged from 09_test.py (confusion matrix only)

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/wrk-vakka/users/mohogaya/caisa/code/src")
from offline_dataloader_128_10mm import OfflineDataset128
from train_resnet_models_step08 import ResNetRegressor, ResNetClassifier

PATCHES_DIR = "/wrk-vakka/users/mohogaya/caisa/data/corrupted_patches_128_10mm"


def plot_confusion_matrix(cm, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    classes = ["Clean", "Mild", "Moderate", "Severe"]
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted", fontsize=16); ax.set_ylabel("True", fontsize=16)
    ax.tick_params(axis='both', labelsize=13)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=16)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_regression_scatter(true_vals, pred_vals, mae, rmse, output_path):
    fig, ax = plt.subplots(figsize=(7, 7))

    #hexbin for density since there are ~19000 points
    hb = ax.hexbin(true_vals, pred_vals, gridsize=40, cmap="Blues", mincnt=1)
    plt.colorbar(hb, ax=ax, fraction=0.046, label="Patch count")

    lo = min(true_vals.min(), pred_vals.min())
    hi = max(true_vals.max(), pred_vals.max())
    ax.plot([lo, hi], [lo, hi], color="red", lw=1.5, ls="--", label="Ideal (y = x)")

    ax.set_xlabel("Simulated severity (mm)", fontsize=16)
    ax.set_ylabel("Predicted severity (mm)", fontsize=16)
    ax.tick_params(axis='both', labelsize=13)
    ax.legend(fontsize=13, loc="upper left")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--patches_dir", type=str, default=PATCHES_DIR)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)

    with open(result_dir / "config.json") as f:
        config = json.load(f)
    mode = config["mode"]
    backbone = config["backbone"]
    print(f"Evaluating: {backbone} {mode} from {result_dir.name}")

    with open(result_dir / "data_splits.json") as f:
        splits = json.load(f)
    test_cases = splits["test"]
    print(f"Test cases: {len(test_cases)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_dataset = OfflineDataset128(args.patches_dir, test_cases, mode)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    if mode == "regression":
        model = ResNetRegressor(backbone)
    else:
        model = ResNetClassifier(backbone, num_classes=4)
    model.load_state_dict(torch.load(result_dir / "best_model.pth", map_location=device))
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if mode == "regression":
                preds = outputs.cpu().numpy()
            else:
                preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    print(f"\nTest Results - {backbone} {mode}")
    print("="*50)

    if mode == "regression":
        mae = float(np.mean(np.abs(all_preds - all_labels)))
        rmse = float(np.sqrt(np.mean((all_preds - all_labels)**2)))
        print(f"MAE:  {mae:.4f} mm")
        print(f"RMSE: {rmse:.4f} mm")

        np.savez(result_dir / "test_predictions.npz", true=all_labels, pred=all_preds)
        plot_regression_scatter(all_labels, all_preds, mae, rmse,
                                result_dir / "test_scatter.png")
        print('Saved: ' + str(result_dir / "test_scatter.png"))

        results = {"mode": mode, "backbone": backbone, "MAE": mae, "RMSE": rmse,
                   "test_cases": len(test_cases), "test_patches": len(test_dataset)}
    else:
        accuracy = float(np.mean(all_preds == all_labels))
        f1 = float(f1_score(all_labels, all_preds, average="weighted"))
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds,
                                       target_names=["Clean", "Mild", "Moderate", "Severe"])
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"\n{report}")
        plot_confusion_matrix(cm, result_dir / "test_confusion_matrix.png")

        results = {"mode": mode, "backbone": backbone, "accuracy": accuracy, "f1": f1,
                   "test_cases": len(test_cases), "test_patches": len(test_dataset)}

    with open(result_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print('Saved to: ' + str(result_dir / "test_results.json"))


if __name__ == "__main__":
    main()
