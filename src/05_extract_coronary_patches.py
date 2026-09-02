#Script to extract 2D patches from Imagecas 3D volumes

"""
To run:

# ex: extract patches from case 50 to case 150 in ImageCAS

python 04_patch_extraction.py --start 50 --end 150

or

python 04_patch_extraction.py --start 1 --end 100 \
    --data-dir "D:\PhD HUS\CAISA\data\datasets\imagecas\niigz" \
    --output-dir "D:\PhD HUS\CAISA\data\patches"

NOTE: replace paths with your own data directories.

"""


import os
import sys
import json
import argparse
import numpy as np
import SimpleITK as sitk
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from scipy.ndimage import distance_transform_edt, maximum_filter, map_coordinates
from scipy.spatial import cKDTree
import warnings

warnings.filterwarnings('ignore')

@dataclass
class ExtractionConfig:
    patch_size: int = 64
    patch_spacing_mm: float = 0.4
    centerline_sampling_mm: float = 2.0
    min_vessel_radius: float = 0.5
    hu_min: float = -250.0
    hu_max: float = 650.0

class SimpleExtractor:
    def __init__(self, config: ExtractionConfig):
        self.config = config

    #Binary mask
    def extract_centerline(self, label_volume: np.ndarray) -> np.ndarray:
        binary = (label_volume > 0).astype(np.uint8)
        if binary.sum() == 0: 
            return np.array([]).reshape(0, 3)
            
        dist = distance_transform_edt(binary)
        local_max = (dist == maximum_filter(dist, size=3)) & (dist > self.config.min_vessel_radius)
        return np.array(np.where(local_max)).T

    #Nearest neighbor greedy search
    def order_points(self, coords: np.ndarray, max_gap: float = 5.0) -> np.ndarray:
        if len(coords) < 2: 
            return coords
            
        tree = cKDTree(coords)
        neighbor_counts = np.array([len(tree.query_ball_point(p, r=2)) for p in coords])
        start_idx = np.argmin(neighbor_counts)
        
        ordered = [coords[start_idx]]
        remaining = set(range(len(coords))) - {start_idx}
        
        while remaining:
            current = ordered[-1]
            nearby_indices = tree.query_ball_point(current, max_gap)
            valid_nearby = [i for i in nearby_indices if i in remaining]
            
            if not valid_nearby:
                break
                
            dists = [np.linalg.norm(coords[i] - current) for i in valid_nearby]
            nearest_idx = valid_nearby[np.argmin(dists)]
            
            ordered.append(coords[nearest_idx])
            remaining.remove(nearest_idx)
            
        return np.array(ordered)
    #compute tangents along the line
    def compute_tangents(self, centerline: np.ndarray) -> np.ndarray:
        tangents = np.zeros_like(centerline, dtype=float)
        n = len(centerline)
        for i in range(n):
            if i == 0: 
                tangents[i] = centerline[min(1, n-1)] - centerline[0]
            elif i == n - 1: 
                tangents[i] = centerline[-1] - centerline[-2]
            else: 
                tangents[i] = centerline[i+1] - centerline[i-1]
                
            norm = np.linalg.norm(tangents[i])
            if norm > 0: 
                tangents[i] /= norm
        return tangents
    
    #Resample centerline
    def sample_points(self, centerline: np.ndarray, voxel_spacing: Tuple[float, float, float]) -> List[int]:
        if len(centerline) < 2: return []
        
        spacing_arr = np.array(voxel_spacing)
        diffs = np.diff(centerline, axis=0) * spacing_arr
        arc_length = np.concatenate([[0], np.cumsum(np.linalg.norm(diffs, axis=1))])
        total_length = arc_length[-1]
        
        n_samples = max(1, int(total_length / self.config.centerline_sampling_mm))
        sample_positions = np.linspace(0, total_length, n_samples)
        
        indices = []
        for pos in sample_positions:
            idx = min(np.searchsorted(arc_length, pos), len(centerline) - 1)
            if idx not in indices: 
                indices.append(idx)
        return indices

    #crop 2D patches perpendiculat rto centerline
    def extract_patch(self, volume: np.ndarray, center: np.ndarray, 
                     tangent: np.ndarray, voxel_spacing: Tuple[float, float, float]) -> np.ndarray:
        patch_size = self.config.patch_size
        output_spacing = self.config.patch_spacing_mm
        
        tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
        v1 = np.array([1.0, 0.0, 0.0]) if abs(tangent[0]) < 0.9 else np.array([0.0, 1.0, 0.0])

        v1 = (v1 - np.dot(v1, tangent) * tangent)
        v1 /= (np.linalg.norm(v1) + 1e-8)
        v2 = np.cross(tangent, v1)
        v2 /= (np.linalg.norm(v2) + 1e-8)

        half = patch_size // 2
        coords_1d = np.arange(-half, half) * output_spacing
        grid_v1, grid_v2 = np.meshgrid(coords_1d, coords_1d, indexing='ij')
        
        spacing_arr = np.array(voxel_spacing)
        sample_coords = np.zeros((3, patch_size, patch_size))
        
        for i in range(3):
            sample_coords[i] = center[i] + (grid_v1 * v1[i] + grid_v2 * v2[i]) / spacing_arr[i]
            
        return map_coordinates(volume, sample_coords, order=1, mode='constant', cval=0).astype(np.float32)

    def normalize_hu(self, volume: np.ndarray) -> np.ndarray:
        clipped = np.clip(volume, self.config.hu_min, self.config.hu_max)
        return (clipped - self.config.hu_min) / (self.config.hu_max - self.config.hu_min)

def process_case(case_id: int, data_dir: str, output_dir: str, config: ExtractionConfig) -> int:
    image_path = os.path.join(data_dir, f"{case_id:04d}.img.nii.gz")
    label_path = os.path.join(data_dir, f"{case_id:04d}.label.nii.gz")
    case_out_dir = os.path.join(output_dir, f"case_{case_id:04d}")
    
    if not os.path.exists(image_path) or not os.path.exists(label_path):
        return 0

    extractor = SimpleExtractor(config)

    try:
        ct_img = sitk.ReadImage(image_path)
        ct_vol = sitk.GetArrayFromImage(ct_img).astype(np.float32)
        label_vol = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        voxel_spacing = tuple(reversed(ct_img.GetSpacing()))

        ct_norm = extractor.normalize_hu(ct_vol)

        raw_coords = extractor.extract_centerline(label_vol)
        if len(raw_coords) < 2: return 0

        centerline = extractor.order_points(raw_coords)
        tangents = extractor.compute_tangents(centerline)
        indices = extractor.sample_points(centerline, voxel_spacing)
        
        os.makedirs(case_out_dir, exist_ok=True)
        count = 0
        
        for i, idx in enumerate(indices):
            patch = extractor.extract_patch(ct_norm, centerline[idx], tangents[idx], voxel_spacing)
            
            if patch.max() == 0: continue
                
            filename = f"patch_{i:03d}.nii.gz"
            save_path = os.path.join(case_out_dir, filename)
            
            img_out = sitk.GetImageFromArray(patch[np.newaxis, :, :])
            img_out.SetSpacing((config.patch_spacing_mm, config.patch_spacing_mm, config.patch_spacing_mm))
            sitk.WriteImage(img_out, save_path)
            
            count += 1
            
        return count

    except Exception as e:
        print(f"Error case {case_id}: {e}")
        return 0

def extract_all_patches(data_dir: str, output_dir: str, start_case: int, end_case: int, config: ExtractionConfig):
    os.makedirs(output_dir, exist_ok=True)
    metadata = {
        'config': {
            'patch_size': config.patch_size,
            'sampling_mm': config.centerline_sampling_mm
        },
        'cases': []
    }
    
    total_patches = 0
    print(f"Starting extraction from case {start_case} to {end_case}...")
    
    for case_id in range(start_case, end_case + 1):
        print(f"Processing case {case_id:04d}...", end=' ')
        n = process_case(case_id, data_dir, output_dir, config)
        
        if n > 0:
            print(f"Saved {n} patches.")
            total_patches += n
            metadata['cases'].append({'case_id': case_id, 'num_patches': n})
        else:
            print("Skipped.")
            
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\nDone. Total patches: {total_patches}")

def main():
    parser = argparse.ArgumentParser()
    #Replace paths
    parser.add_argument('--data-dir', type=str, default=r"/wrk-vakka/users/mohogaya/caisa/data/datasets/imagecas/niigz")
    parser.add_argument('--output-dir', type=str, default=r"/wrk-vakka/users/mohogaya/caisa/data/datasets/imagecas/niigz")
    #parser.add_argument('--data-dir', type=str, default=r"D:\PhD HUS\CAISA\data\datasets\imagecas\niigz")
    #parser.add_argument('--output-dir', type=str, default=r"D:\PhD HUS\CAISA\data\patches")
    parser.add_argument('--start',type=int, default=1, help='Start case ID' )
    parser.add_argument('--end',type=int, default=1000, help='End case ID' )
    parser.add_argument('--num-cases', type=int, default=None, help='(Deprecated) Use --start and --end instead')
    parser.add_argument('--patch-size', type=int, default=64)
    args = parser.parse_args()
    
    config = ExtractionConfig(patch_size=args.patch_size)

    if args.num_cases is not None:
        print("Warning: --num-cases is deprecated. Use --start and --end instead.")
        start = 1
        end = args.num_cases
    else:
        start = args.start
        end = args.end
    
    extract_all_patches(args.data_dir, args.output_dir, start, end, config) 


if __name__ == "__main__":
    main()
