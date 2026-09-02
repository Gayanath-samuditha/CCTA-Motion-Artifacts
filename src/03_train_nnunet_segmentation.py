# nnUNet training script

import os
import subprocess
import sys
from pathlib import Path

from local_env import set_environment_variables
set_environment_variables()

def setup_nnunet_paths():  
    #Configure nnU-net variables

    base_dir = Path(os.environ["CAISA_DATA_FOLDER"])
    
    os.environ["nnUNet_raw"] = str(base_dir / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(base_dir / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(base_dir / "nnUNet_results")
    
    print("nnUNet paths:")
    print(f"  Raw: {os.environ['nnUNet_raw']}")
    print(f"  Preprocessed: {os.environ['nnUNet_preprocessed']}")
    print(f"  Results: {os.environ['nnUNet_results']}")

def detect_device():
    #Detect availavle compute device
    #Works for cloud clusters too
    
    try:
        import torch #Use torch 2.8.0 or lower
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Using CUDA: {gpu_name}")
        else:
            device = "cpu"
            print("Using CPU")
    except ImportError:
        device = "cpu"
        print("Pytorch not available. Defaulting to CPU!")
    
    return device


def sanity_check():
    #Quick test run to verify nnU-Net installation.
    
    
    print("\nTest Run")

    import os
    os.environ['nnUNet_n_proc_DA'] = '2'  #2 workers instead of 8
    preprocessed_dir = Path(os.environ["nnUNet_preprocessed"]) / "Dataset001_ImageCAS"

    if not preprocessed_dir.exists():
        print("\n Preprocessing required. Running now...")
        try:
            from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess_entry
            
            import sys
            old_argv = sys.argv
            sys.argv = ['plan_and_preprocess', '-d', '1','-np', '2']
            
            plan_and_preprocess_entry()
            
            sys.argv = old_argv
            print("\n Preprocessing complete!")
        except Exception as e:
            print(f"\n Preprocessing failed: {e}")
            print("Check your data in nnUNet_raw/Dataset001_ImageCAS")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("\n Preprocessed data found. Skipping preprocessing.")
    
    print("\n Running 2-epoch test on 3d_fullres fold 0...")
    
    try:
        import time
        from nnunetv2.run.run_training import run_training_entry
        
        import sys
        old_argv = sys.argv
        sys.argv = [
            'nnUNetv2_train',
            '1',              #dataset_id
            '3d_fullres',     #configuration
            '0',              #fold
            '--npz',
            '-device', 'cuda'
        ]
        
        start_time = time.time()
        
        run_training_entry()
        
        sys.argv = old_argv
        
        elapsed_time = time.time() - start_time
        time_per_epoch = elapsed_time / 2
        estimated_full = (time_per_epoch * 1000 / 3600)  # 1000 epochs in hours
        
        print("TEST RUN COMPLETE")
        print(f"  ~{estimated_full:.1f} hours")
        print(f"\nEstimated full training (1000 epochs, 5 folds):")
        print(f"  ~{estimated_full * 5:.1f} hours")
        
        return True

    except Exception as e:
        print(f"\n Test run failed: {e}")
        print("Check error messages above for details.")
        import traceback
        traceback.print_exc()
        return False
    

def run_planning_and_preprocessing(configurations=None):
    #This analyzes dataset properties
    #Determines patch size, batch size, normalization

    # Args: Specific configuration to plan
    import os
    os.environ['nnUNet_n_proc_DA']='2'
    print("\nPlanning and preprocessing ImageCAS")

    cmd =[
        "nnUNetv2_plan_and_preprocess",
        "-d", "1",
        "-np", "2",
        "-c", "3d_lowres", "3d_fullres" 
    ]

    if configurations:
        cmd.extend(["-c"] + configurations)
    
    subprocess.run(cmd, check=True)
    print("\nPlanning Done!.")

def train_single_fold(configuration, fold, device='cuda'):
    #Train one fold of a configuration

    #Args: configuration will be 2D or 3D_fullres
    #      fold: 0-4 (5 fold cross validation)
    
    cmd = [
        "nnUNetv2_train",
        "1",
        configuration,
        str(fold),
        "--npz",
        "-device", device
    ]
    
    print(f"Training {configuration} fold {fold}...")
    subprocess.run(cmd, check=True)

def train_configuration(configuration, device="cuda", parallel=False):
    #Train all 5 folds of  a configuration
    #This has both seuquential and parallel modes(In case if have multiple GPUs from a cluster)
    #Args: config: 3D_lowres or 3D_fullres
    #      device: cuda, cpu
    #      Parallel: If true, only trains fold 0 and prints parallel commands
            
    print(f"\nTRAINING: {configuration}")

    # Always train fold 0 first (hence preprocessing)
    train_single_fold(configuration, 0, device)

    if parallel:
        print(f"\nFold 0 complete. For parallel training on cluster:")
        print(f"Run these commands simultaneously on separate GPUs:\n")
        for fold in range(1, 5):
            print(f"CUDA_VISIBLE_DEVICES={fold-1} nnUNetv2_train 1 {configuration} {fold} --npz -device cuda &")
        print("\nwait")
        print(f"\nExiting after fold 0. Run remaining folds on cluster.")
        return
    
    #Sequential training
    for fold in range(1, 5):
        train_single_fold(configuration, fold, device)
    
    print(f"\n{configuration}: All 5 folds complete.")

def find_best_configuration(configurations):
    # Determines best configuration for ImageCAS

    print("\nFINDING BEST CONFIGURATION")

    configs_str = " ".join(configurations)
    
    cmd = [
        "nnUNetv2_find_best_configuration",
        "1",
        "-c"
    ] + configurations
    
    print(f"\nEvaluating: {configs_str}")
    print("This determines optimal configuration and postprocessing.\n")
    
    subprocess.run(cmd, check=True)
    
    results_dir = Path(os.environ["nnUNet_results"]) / "Dataset001_ImageCAS"
    print("\n Results saved:")
    print(f"  {results_dir}/inference_instructions.txt")
    print(f"  {results_dir}/inference_information.json")
    
def train_all_configurations(device="cuda", parallel=False):

    #Train multiple configurations

    configs = ["3d_lowres", "3d_fullres"]
    
    for config in configs:
        train_configuration(config, device, parallel)
        
        if parallel and config == configs[0]:
            print("\nParallel mode: Complete remaining folds on cluster,")
            print("then run this script again with configurations already trained.")
            sys.exit(0)
    
    return configs

def main():
    """
    Complete training pipeline for ImageCAS.
    
    Set parallel=True for cloud clusters with multiple GPUs.
    This trains fold 0, then provides commands for parallel execution.
    
    steps:
    1. Plan and preprocess
    2. Train all configurations
    3. Find best configuration
    4. Generate inference instructions
    """
    PARALLEL_MODE = False  #Set True for kumpula cluster with multiple GPUs
    TEST_RUN_ONLY = False #Set True for quick test run
    
    print("\nnnU-Net TRAINING: IMAGECAS")
    print("\nDataset: ImageCAS CCTA")
    print(" Task: Coronary artery segmentation")
    print(" Training: 800 cases")
    print(" Test: 200 cases")
    
    if TEST_RUN_ONLY:
        print(f"Test run for 2 epochs")
        setup_nnunet_paths()
        detect_device()

        success=sanity_check()

        if success:
            print("Test run passed")
            print("set TEST_RUN_ONLY=False to run full training")
        else:
            print("Test run failed. check for issues")
        return

    if PARALLEL_MODE:
        print("\nMode:PARALLEL")
        print(" Will train fold 0, then exit for parallel execution")
    else:
        print("\nMode:Single GPU")
        print("  All folds trained sequentially")
    
    #Setup
    print("\nEnvironment setup")
    setup_nnunet_paths()
    device = detect_device()
    
    #Planning
    print("\nPlanning and preprocessing")
    run_planning_and_preprocessing()
    
    #Training
    print("\ntraining configurations")
    configs = train_all_configurations(device, PARALLEL_MODE)
    
    #Find best
    print("\nFinding best configuration")
    find_best_configuration(configs)
    
    #Summary
    print("\nTRAINING COMPLETE!")
    results_dir = Path(os.environ["nnUNet_results"]) / "Dataset001_ImageCAS"
    print(f"\nResults: {results_dir}")

if __name__ == '__main__':
    main()