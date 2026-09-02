# CAISA: Automated CCTA Image Quality Assessment

Code for the paper "Simulation and Quantitative Assessment of Coronary Motion Artifacts and Contrast
Enhancement in CCTA" (Computerized Medical Imaging and Graphics, not published yet).

## Pipeline order

1. `01_download_imagecas.py` - download/organise the ImageCAS dataset
2. `02_convert_to_nnunet_format.py` - convert to nnU-Net directory structure
3. `03_train_nnunet_segmentation.py` - train nnU-Net for coronary segmentation
4. `04_run_totalsegmentator_inference.py` - run TotalSegmentator inference on all volumes
5. `05_extract_coronary_patches.py` - extract 128x128 coronary cross-sectional patches
6. `sinogram_motion_corruption_step06.py` - sinogram-based motion corruption (imported by step 7)
7. `07_generate_corrupted_dataset.py` - generate corrupted patches at 11 severity levels (0-10 mm)
8. `train_resnet_models_step08.py` - train ResNet classification/regression models (imported by step 9)
9. `09_evaluate_test_set.py` - evaluate trained models on the held-out test set
10. `10_segment_septum.py` - segment the interventricular septum
11. `11_compute_cnr.py` - compute CNR (signal, background, noise) per case
12. `12_plot_cnr_histogram.py` - plot the CNR distribution figure

## Supporting utilities

- `local_env.py` - sets environment variables used by the data download/conversion steps
- `data_download.py` - downloads the ImageCAS dataset from Kaggle
- `offline_dataloader_128_10mm.py` - dataset class used during evaluation (step 9)

## Dataset

Requires the publicly available ImageCAS dataset (Zeng et al., 2023),
not included here due to size and licensing.

## Requirements

See requirements.txt. Also requires the ASTRA Toolbox and TotalSegmentator.

## TotalSegmentator license

Step 4 requires a TotalSegmentator license key. Set it via the
`TOTALSEG_LICENSE` environment variable, or edit the placeholder in
`04_run_totalsegmentator_inference.py` directly.

## Note on paths

Scripts contain hardcoded paths from the original compute environment
(Turso HPC, University of Helsinki). Update `patches_dir`, `output_dir`,
and similar variables at the top of each script before running.

