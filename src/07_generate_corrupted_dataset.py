"""
Generate corrupted 128x128 patches, 0-10mm severity, random direction, amp=1.0.
Usage: python generate_corrupted_128_10mm.py
"""

import os
import argparse
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from sinogram_motion_corruption_step06 import InterpolationCorruptor, Config

PATCHES_DIR = "/wrk-vakka/users/mohogaya/caisa/data/patches_128"
OUTPUT_DIR = "/wrk-vakka/users/mohogaya/caisa/data/corrupted_patches_128_10mm"
SEVERITY_LEVELS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

def process_patch(patch_path, output_dir, corruptor):
    try:
        img = sitk.ReadImage(str(patch_path))
        clean_patch = sitk.GetArrayFromImage(img)[0]
    except Exception as e:
        print(f"ERROR loading {patch_path}: {e}")
        return 0

    patch_name = f"{patch_path.parent.name}_{patch_path.name.replace('.nii.gz', '')}"
    count = 0

    #save clean patch
    out_path = os.path.join(output_dir, f"{patch_name}_sev0.00_class0.nii.gz")
    if not os.path.exists(out_path):
        img_out = sitk.GetImageFromArray(clean_patch[np.newaxis, :, :])
        img_out.SetSpacing((0.4, 0.4, 0.4))
        sitk.WriteImage(img_out, out_path)
    count += 1

    #save corrupted patches
    for sev in SEVERITY_LEVELS:
        if sev == 0.0:
            continue
        try:
            corrupted, meta = corruptor.corrupt(clean_patch, sev)
            cls = meta['severity_class']
            actual_sev = meta['severity']
            out_path = os.path.join(output_dir, f"{patch_name}_sev{actual_sev:.2f}_class{cls}.nii.gz")
            if not os.path.exists(out_path):
                img_out = sitk.GetImageFromArray(corrupted[np.newaxis, :, :])
                img_out.SetSpacing((0.4, 0.4, 0.4))
                sitk.WriteImage(img_out, out_path)
            count += 1
        except Exception as e:
            print(f"ERROR corrupting {patch_path} sev={sev}: {e}")

    return count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patches_dir', type=str, default=PATCHES_DIR)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    #amp=1.0 is default in Config
    config = Config(motion_amplification=1.0)
    corruptor = InterpolationCorruptor(config)

    patch_paths = sorted(Path(args.patches_dir).rglob("patch_*.nii.gz"))
    print(f"Found {len(patch_paths)} patches")
    print(f"Output: {args.output_dir}")

    total = 0
    for patch_path in tqdm(patch_paths):
        total += process_patch(patch_path, args.output_dir, corruptor)

    print(f"Done. Total files saved: {total}")

if __name__ == '__main__':
    main()
