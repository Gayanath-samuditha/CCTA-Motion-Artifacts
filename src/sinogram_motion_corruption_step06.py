
"""
interpolation_05.py - Interpolation-based Motion Artifact Corruption

Generates artifacts by simulating object movement.

Try using different values for parameters in Config for much realistic artifacts.

Run this for sanity check:

python interpolation_05.py --sanity --start 800 --end 850

"""

import os
import argparse
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from scipy.ndimage import map_coordinates, gaussian_filter, distance_transform_edt, binary_closing
import SimpleITK as sitk

try:
    import astra
    CUDA_AVAILABLE = astra.use_cuda()
except ImportError:
    raise ImportError("ASTRA required: conda install -c astra-toolbox astra-toolbox")


@dataclass
class Config:
    #CT geometry
    num_angles: int = 180  
    pixel_spacing: float = 0.4  #Range:0.3-0.6mm(patch resolution)
    
    #Motion parameters
    motion_range_mm: Tuple[float, float] = (0.0, 10.0)  #Range: severity
    dilation_mm: float = 3  #Range:3-10mm(motion spread from vessel)
    motion_amplification: float = 1.0 #Reduced for 64x64 patches with 0-10mm range
    
    #Patch processing
    padding: int = 32  
    blur_sigma: float = 0.1   #Range:0.3-0.6(post-recon smoothing)


class InterpolationCorruptor:
    """Generate motion artifacts via progressive sinogram interpolation"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
    
    def _create_vessel_mask(self, patch: np.ndarray, threshold: float = 0.15) -> np.ndarray:
        #Create vessel mask from intensity
        normalized = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
        mask = (normalized > threshold).astype(np.uint8)
        mask = binary_closing(mask, structure=np.ones((3, 3)))
        return mask
    
    def _create_displacement_field(self, mask: np.ndarray, dir_x: float, dir_y: float) -> Tuple[np.ndarray, np.ndarray]:
        #Create Gaussian displacement field from vessel mask
        h, w = mask.shape
        
        if mask.sum() == 0:
            return np.zeros((h, w)), np.zeros((h, w))
        
        #Distance transform
        dist = distance_transform_edt(~mask.astype(bool))
        dilation_px = self.config.dilation_mm / self.config.pixel_spacing
        sigma_px = dilation_px / 2
        magnitude = np.exp(-dist**2 / (2 * sigma_px**2))
        magnitude[mask > 0] = 1.0
        
        #Directional displacement
        disp_x = magnitude * dir_x
        disp_y = magnitude * dir_y
        
        return disp_x, disp_y
    
    def _apply_deformation(self, image: np.ndarray, disp_x: np.ndarray, 
                          disp_y: np.ndarray, factor: float) -> np.ndarray:
        #Apply scaled displacement field
        h, w = image.shape
        yy, xx = np.mgrid[:h, :w]
        
        new_y = yy + disp_y * factor
        new_x = xx + disp_x * factor
        new_y = np.clip(new_y, 0, h - 1)
        new_x = np.clip(new_x, 0, w - 1)
        
        deformed = map_coordinates(image, [new_y, new_x], order=1, mode='nearest')
        return deformed.astype(np.float32)
    
    def _get_astra_geometry(self, h: int, w: int, angles: np.ndarray):
        #ASTRA parallel-beam geometry
        vol_geom = astra.create_vol_geom(h, w)
        nr_detectors = max(h, w) + 64
        proj_geom = astra.create_proj_geom('parallel', 1.0, nr_detectors, angles)
        return vol_geom, proj_geom, nr_detectors
    
    def corrupt(self, patch: np.ndarray, severity: float, 
                direction: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
        
        #sseverity
        severity = float(np.clip(severity, 0.0, 10.0))
        
        #Clean case
        if severity < 0.01:
            return patch.copy(), {
                'severity': 0.0,
                'severity_class': 0,
                'direction': 0.0,
                'is_clean': True
            }
        
        #Random direction if not specified
        if direction is None:
            direction = np.random.uniform(0, 2 * np.pi)
        
        pad = self.config.padding
        patch_padded = np.pad(patch, pad, mode='edge')
        h, w = patch_padded.shape
        
        #Create vessel mask
        vessel_mask = self._create_vessel_mask(patch_padded)
        
        #Convert severity to pixel displacement
        magnitude_px = (severity * self.config.motion_amplification) / self.config.pixel_spacing
        dir_x = magnitude_px * np.cos(direction)
        dir_y = magnitude_px * np.sin(direction)
        
        #Create displacement field
        disp_x, disp_y = self._create_displacement_field(vessel_mask, dir_x, dir_y)
        
        #Setup ASTRA
        angles = np.linspace(0, np.pi, self.config.num_angles, endpoint=False)
        vol_geom, proj_geom, nr_detectors = self._get_astra_geometry(h, w, angles)
        
        #Initialize sinogram
        sinogram = np.zeros((self.config.num_angles, nr_detectors))
        
        #Progressive motion schedule
        max_disp = np.sqrt(dir_x**2 + dir_y**2)
        n_motion_steps = max(8, int(max_disp * 0.5))
        motion_angles = np.linspace(0, self.config.num_angles - 1, n_motion_steps).astype(int)
        
        #Progressive deformation factor
        cumulative_factor = 0.0
        step_size = 1.0 / (n_motion_steps - 1) if n_motion_steps > 1 else 1.0
        motion_step_idx = 0
        
        try:
            #For each projection angle
            for i, angle in enumerate(angles):
                #Update motion factor at scheduled angles
                if motion_step_idx < len(motion_angles) and i >= motion_angles[motion_step_idx]:
                    cumulative_factor = min(1.0, (motion_step_idx + 1) * step_size)
                    motion_step_idx += 1
                
                #Deform image at current motion stage
                deformed = self._apply_deformation(patch_padded, disp_x, disp_y, cumulative_factor)
                
                #Forward project this deformed image
                proj_geom_single = astra.create_proj_geom('parallel', 1.0, nr_detectors, [angle])
                projector_id = astra.create_projector('cuda' if CUDA_AVAILABLE else 'line', 
                                                      proj_geom_single, vol_geom)
                
                sino_id, sino_line = astra.creators.create_sino(deformed, projector_id, returnData=True)
                sinogram[i, :] = sino_line
                
                #Cleanup
                astra.data2d.delete(sino_id)
                astra.projector.delete(projector_id)
            
            #Reconstruct from corrupted sinogram
            projector_id = astra.create_projector('cuda' if CUDA_AVAILABLE else 'line',
                                                  proj_geom, vol_geom)
            
            sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
            reconstruction_id = astra.data2d.create('-vol', vol_geom, data=0)
            
            alg_cfg = astra.astra_dict('FBP_CUDA' if CUDA_AVAILABLE else 'FBP')
            alg_cfg['ProjectorId'] = projector_id
            alg_cfg['ProjectionDataId'] = sinogram_id
            alg_cfg['ReconstructionDataId'] = reconstruction_id
            
            algorithm_id = astra.algorithm.create(alg_cfg)
            astra.algorithm.run(algorithm_id)
            recon = astra.data2d.get(reconstruction_id)
            
            #Cleanup
            astra.algorithm.delete(algorithm_id)
            astra.data2d.delete(sinogram_id)
            astra.data2d.delete(reconstruction_id)
            astra.projector.delete(projector_id)
        
        except Exception as e:
            print(f"ASTRA error: {e}")
            return patch.copy(), {
                'severity': 0.0,
                'severity_class': 0,
                'direction': 0.0,
                'is_clean': True,
                'error': str(e)
            }
        
        recon = gaussian_filter(recon, sigma=self.config.blur_sigma)
        
        #Crop to original size
        result = recon[pad:-pad, pad:-pad]
        
        #intensity range
        result = np.clip(result, patch.min(), patch.max()).astype(np.float32)
        
        #Severity classification (10 classes for 0.0-2.0mm in 0.2mm steps)
        #sev_class = int(round(severity / 0.2))
        #sev_class = np.clip(sev_class, 0, 10)

        if severity < 0.01:
            sev_class = 0
        elif severity < 2.5:
            sev_class = 1
        elif severity < 6.0:
            sev_class = 2
        else:
            sev_class = 3


        
        metadata = {
            'severity': float(severity),
            'severity_class': int(sev_class),
            'direction': float(direction),
            'is_clean': False
        }
        
        return result, metadata
    
    def corrupt_random(self, patch: np.ndarray) -> Tuple[np.ndarray, Dict]:
        #Corrupt with random severity 0.0-2.0
        min_sev = float(self.config.motion_range_mm[0])
        max_sev = float(self.config.motion_range_mm[1])
        severity = np.random.uniform(min_sev, max_sev)
        return self.corrupt(patch, severity)


#Utility functions

def load_patch(patches_dir: str, case_id: int, patch_idx: int) -> np.ndarray:
    #Load single patch from disk
    case_dir = os.path.join(patches_dir, f"case_{case_id:04d}")
    patch_path = os.path.join(case_dir, f"patch_{patch_idx:03d}.nii.gz")
    
    if not os.path.exists(patch_path):
        raise FileNotFoundError(f"Patch not found: {patch_path}")
    
    img = sitk.ReadImage(patch_path)
    return sitk.GetArrayFromImage(img)[0]


def load_all_patches(patches_dir: str, case_ids: list) -> list:
    #Load all patches from specified cases
    patches = []
    for cid in case_ids:
        case_dir = os.path.join(patches_dir, f"case_{cid:04d}")
        if not os.path.exists(case_dir):
            continue
        
        for fname in sorted(os.listdir(case_dir)):
            if fname.endswith('.nii.gz'):
                path = os.path.join(case_dir, fname)
                try:
                    img = sitk.ReadImage(path)
                    patches.append(sitk.GetArrayFromImage(img)[0])
                except:
                    continue
    
    return patches


#Testing and visualization

def test():
    #Test basic functionality
    print(f"ASTRA CUDA: {'enabled' if CUDA_AVAILABLE else 'disabled'}")
    print("Testing InterpolationCorruptor...")
    
    corruptor = InterpolationCorruptor()
    
    print("Loading test patches...")
    patches = load_all_patches("/wrk-vakka/users/mohogaya/caisa/results/patches", [1])
    if len(patches) == 0:  #D:/PhD HUS/CAISA/data/patches
        print("Error: No patches found for testing")
        return
    
    patch = patches[0]
    
    print("Testing severity levels:")
    for sev in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
        out, meta = corruptor.corrupt(patch, sev, direction=0)
        print(f"  sev={sev:.1f}mm: class={meta['severity_class']}, clean={meta['is_clean']}")
    
    print("Test passed")


def visualize(patches_dir: str, num_samples: int):
    #Visualize corruption on multiple patches
    import matplotlib.pyplot as plt
    
    patches = load_all_patches(patches_dir, list(range(1, 20)))
    
    if len(patches) == 0:
        print("Error: No patches found")
        return
    
    indices = np.random.choice(len(patches), min(num_samples, len(patches)), replace=False)
    
    corruptor = InterpolationCorruptor()
    severities = [0.0, 0.4, 1.0, 2.0]
    
    fig, axes = plt.subplots(num_samples, len(severities), 
                            figsize=(3*len(severities), 3*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for row, idx in enumerate(indices):
        for col, sev in enumerate(severities):
            corrupted, _ = corruptor.corrupt(patches[idx], sev, direction=0)
            axes[row, col].imshow(corrupted, cmap='gray')
            if row == 0:
                axes[row, col].set_title(f"{sev:.1f}mm")
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig("interpolation_visualization.png", dpi=150)
    plt.show()
    print("Saved: interpolation_visualization.png")


def single(patches_dir: str, case_id: int, patch_idx: int, severity: float):
    #Compare clean vs corrupted
    import matplotlib.pyplot as plt
    
    patch = load_patch(patches_dir, case_id, patch_idx)
    corruptor = InterpolationCorruptor()
    
    clean = patch.copy()
    corrupted, meta = corruptor.corrupt(patch, severity, direction=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    axes[0].imshow(clean, cmap='gray')
    axes[0].set_title("Clean")
    axes[0].axis('off')
    
    axes[1].imshow(corrupted, cmap='gray')
    axes[1].set_title(f"{meta['severity']:.1f}mm (class {meta['severity_class']})")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("interpolation_single.png", dpi=150)
    plt.show()
    print("Saved: interpolation_single.png")


def sanity_check(patches_dir: str, output_dir: str, start_case: int, end_case: int, 
                 patches_per_case: int = 10):
    #Generate corrupted patches with 0.2mm severity gaps from 0.2-2.0mm
    os.makedirs(output_dir, exist_ok=True)
    
    corruptor = InterpolationCorruptor()
    spacing = 0.4
    
    #Generate severity levels: 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0
    severity_levels = [round(i * 0.2, 1) for i in range(1, 11)]
    
    print(f"\n{'='*70}")
    print("SANITY CHECK MODE - Interpolation Method")
    print(f"{'='*70}")
    print(f"Cases: {start_case} to {end_case}")
    print(f"Patches/case: {patches_per_case}")
    print(f"Severity levels: {severity_levels} mm")
    print(f"Total patches per input: 10 (9 corrupted + 1 clean)")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    total_saved = 0
    
    for case_id in range(start_case, end_case + 1):
        case_dir = os.path.join(patches_dir, f"case_{case_id:04d}")
        if not os.path.exists(case_dir):
            print(f"Case {case_id:04d}: Not found, skipping")
            continue
        
        patch_files = sorted([f for f in os.listdir(case_dir) if f.endswith('.nii.gz')])
        if len(patch_files) == 0:
            print(f"Case {case_id:04d}: No patches, skipping")
            continue
        
        selected = np.random.choice(
            min(len(patch_files), patches_per_case), 
            min(patches_per_case, len(patch_files)), 
            replace=False
        )
        
        case_out_dir = os.path.join(output_dir, f"case_{case_id:04d}")
        os.makedirs(case_out_dir, exist_ok=True)
        
        print(f"Case {case_id:04d}: Processing {len(selected)} patches...")
        
        for patch_idx in selected:
            patch_path = os.path.join(case_dir, patch_files[patch_idx])
            img = sitk.ReadImage(patch_path)
            clean_patch = sitk.GetArrayFromImage(img)[0]
            
            #Save clean
            out_name = f"patch_{patch_idx:03d}_sev0.0mm_class0.nii.gz"
            out_path = os.path.join(case_out_dir, out_name)
            img_out = sitk.GetImageFromArray(clean_patch[np.newaxis, :, :])
            img_out.SetSpacing((spacing, spacing, spacing))
            sitk.WriteImage(img_out, out_path)
            total_saved += 1
            
            #Generate all 9 severity levels
            for sev in severity_levels:
                corrupted, meta = corruptor.corrupt(clean_patch, sev, direction=0)
                
                out_name = f"patch_{patch_idx:03d}_sev{sev:.1f}mm_class{meta['severity_class']}.nii.gz"
                out_path = os.path.join(case_out_dir, out_name)
                
                img_out = sitk.GetImageFromArray(corrupted[np.newaxis, :, :])
                img_out.SetSpacing((spacing, spacing, spacing))
                sitk.WriteImage(img_out, out_path)
                total_saved += 1
        
        print(f"  Saved {len(selected) * 10} files")
    
    print(f"\n{'='*70}")
    print(f"Complete! Total files: {total_saved}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Interpolation Motion Corruption')
    parser.add_argument('--test', action='store_true', help='Run basic tests')
    parser.add_argument('--visualize', action='store_true', help='Visualize on patches')
    parser.add_argument('--single', action='store_true', help='Test single patch')
    parser.add_argument('--sanity', action='store_true', help='Sanity check mode')
    
    #parser.add_argument('--patches-dir', type=str, default="D:/PhD HUS/CAISA/data/patches")
    parser.add_argument('--patches-dir', type=str, default="/wrk-vakka/users/mohogaya/caisa/results/patches")


    parser.add_argument('--num-samples', type=int, default=4)
    parser.add_argument('--case', type=int, default=1)
    parser.add_argument('--patch', type=int, default=0)
    parser.add_argument('--severity', type=float, default=1.0)
    
    parser.add_argument('--start', type=int, default=1, help='Start case ID')
    parser.add_argument('--end', type=int, default=3, help='End case ID')
    parser.add_argument('--sanity-output', type=str, default="D:/PhD HUS/CAISA/data/sanity_check_interp")

    parser.add_argument('--sanity-patches', type=int, default=10, help='Patches per case')
    
    parser.add_argument('--num-angles', type=int, default=180, help='Projection angles (90-360)')
    parser.add_argument('--dilation-mm', type=float, default=1.0, help='Motion spread (1-10mm)')
    parser.add_argument('--blur-sigma', type=float, default=0.1, help='Smoothing (0.1-0.6)')
    parser.add_argument('--motion-amp', type=float, default=7.0, help='Amplification (1.5-10)')
    
    args = parser.parse_args()
    
    config = Config(
        num_angles=args.num_angles,
        dilation_mm=args.dilation_mm,
        blur_sigma=args.blur_sigma,
        motion_amplification=args.motion_amp
    )
    
    global_corruptor = InterpolationCorruptor(config)
    
    if args.test:
        test()
    elif args.visualize:
        visualize(args.patches_dir, args.num_samples)
    elif args.single:
        single(args.patches_dir, args.case, args.patch, args.severity)
    elif args.sanity:
        sanity_check(
            patches_dir=args.patches_dir,
            output_dir=args.sanity_output,
            start_case=args.start,
            end_case=args.end,
            patches_per_case=args.sanity_patches
        )
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
