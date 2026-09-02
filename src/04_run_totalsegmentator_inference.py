#!/usr/bin/env python3
"""
TotalSegmentator inference script for Turso cluster
Segments coronary arteries and heart chambers from ImageCAS dataset
"""

import os
import subprocess
from pathlib import Path


def setup_TotalSegmentator_license():
    """Setup TotalSegmentator license.
    Obtain a license key and set it via the TOTALSEG_LICENSE environment variable, or
    replace the placeholder below before running."""
    import os
    license_key = os.environ.get("TOTALSEG_LICENSE", "YOUR_LICENSE_KEY_HERE")
    
    cmd = ["totalseg_set_license", "-l", license_key]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print("TotalSegmentator License Activated")
    except subprocess.CalledProcessError:
        print("License already set or error occurred!")


def get_test_images():
    """Get list of CT images to process"""
    #Cluster path to ImageCAS CT scans
    test_dir = Path("/wrk-vakka/users/mohogaya/caisa/data/datasets/imagecas/niigz")
    
    if not test_dir.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_dir}")
    
    #images = sorted(test_dir.glob("*.nii.gz"))
    #print(f"Found {len(images)} images")
    #return images
    all_files = sorted(test_dir.glob("*.nii.gz"))
    img_files = [f for f in all_files if f.name.endswith('.img.nii.gz')]
    
    #Filter to cases 1-800 only
    images = [img for img in img_files if int(img.name.split('.')[0]) <= 800]
    
    print(f"Found {len(images)} images (cases 1-800)")
    return images


def setup_output_directory(task_name):
    """Create output directory for TotalSegmentator results"""
    output_dir = Path("/wrk-vakka/users/mohogaya/caisa/data/totalsegmentator_results") / task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory for {task_name}: {output_dir}")
    return output_dir


def run_TotalSegmentator(input_file, output_dir, task):
    """
    Run TotalSegmentator on single image
    
    Args:
        input_file: Path to input NIfTI image
        output_dir: Directory for output segmentation
        task: Task name (coronary_arteries or heartchambers_highres)
    """
    cmd = [
        "TotalSegmentator",
        "-i", str(input_file),
        "-o", str(output_dir),
        "-ta", task,
        #"--fast"  #Uncomment for faster inference
    ]
    
    subprocess.run(cmd, check=True)


def process_task(images, output_dir, task_name, start_idx=0, end_idx=None):
    """
    Process images with TotalSegmentator
    
    Args:
        images: List of image paths
        output_dir: Output directory
        task_name: Task name
        start_idx: Starting index (for parallel processing)
        end_idx: Ending index (for parallel processing)
    """
    if end_idx is None:
        end_idx = len(images)
    
    images_subset = images[start_idx:end_idx]
    failed_cases = []
    
    for idx, image in enumerate(images_subset, start_idx + 1):
        case_name = image.stem.replace('_0000', '')  #Remove _0000 suffix if present
        case_output = output_dir / case_name
        
        print(f"\n[{idx}/{end_idx}] Processing {case_name}")
        
        try:
            run_TotalSegmentator(image, case_output, task_name)
            print(f"  ✓ Complete")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed: {e}")
            failed_cases.append(case_name)
    
    return failed_cases


def print_task_summary(task_name, output_dir, failed_cases, total_cases):
    """Print summary of processing results"""
    success_count = total_cases - len(failed_cases)
    
    print(f"\n{'='*60}")
    print(f"{task_name.upper()} SUMMARY")
    print(f"{'='*60}")
    print(f"Total:   {total_cases}")
    print(f"Success: {success_count}")
    print(f"Failed:  {len(failed_cases)}")
    
    if failed_cases:
        print(f"\nFailed Cases:")
        for case in failed_cases:
            print(f"  - {case}")
    
    print(f"\nOutput: {output_dir}")
    print(f"{'='*60}\n")


def main():
    """
    Run TotalSegmentator inference on ImageCAS dataset
    
    Set to True to enable each task
    """
    RUN_CORONARY_ARTERIES = False
    RUN_HEART_CHAMBERS = True
    
    if not RUN_CORONARY_ARTERIES and not RUN_HEART_CHAMBERS:
        print("Error: At least one task must be enabled!")
        return
    
    print("="*60)
    print("TotalSegmentator Inference on ImageCAS - Turso Cluster")
    print("="*60)
    
    print("\nEnabled tasks:")
    if RUN_CORONARY_ARTERIES:
        print("  ✓ coronary_arteries")
    if RUN_HEART_CHAMBERS:
        print("  ✓ heartchambers_highres")
    
    #Setup license
    #print("\nActivating TotalSegmentator license...")
    #setup_TotalSegmentator_license()
    
    #Get images
    images = get_test_images()[1:800]
    
    #For testing, process subset (remove for full run)
    #images = images[:10]
    
    results = {}
    
    #Coronary arteries
    if RUN_CORONARY_ARTERIES:
        task_name = "coronary_arteries"
        output_dir = setup_output_directory(task_name)
        failed = process_task(images, output_dir, task_name)
        results[task_name] = {
            'output_dir': output_dir,
            'failed': failed
        }
    
    #Heart chambers
    if RUN_HEART_CHAMBERS:
        task_name = "heartchambers_highres"
        output_dir = setup_output_directory(task_name)
        failed = process_task(images, output_dir, task_name)
        results[task_name] = {
            'output_dir': output_dir,
            'failed': failed
        }
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE!")
    print("="*60)
    
    #Print summaries
    for task_name, task_results in results.items():
        print_task_summary(
            task_name,
            task_results['output_dir'],
            task_results['failed'],
            len(images)
        )


if __name__ == '__main__':
    main()
