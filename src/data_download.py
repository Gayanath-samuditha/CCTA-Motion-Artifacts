import os
import shutil
import subprocess
from pathlib import Path

import kagglehub
from tqdm import tqdm

from local_env import set_environment_variables

set_environment_variables()

def download_imagecas_dataset():
    """
    Download the ImageCAS dataset.

    Source:
        - GitHub: https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT
        - Kaggle: https://www.kaggle.com/datasets/xiaoweixumedicalai/imagecas
    """

    kagglehub_cache_dir = Path(os.environ["KAGGLEHUB_CACHE"])
    kagglehub_cache_dir.mkdir(parents=True, exist_ok=True)

    # Download latest version
    path = kagglehub.dataset_download("xiaoweixumedicalai/imagecas")

    print("Downloaded to:", path)

def extract_imagecas_dataset():
    """
    Extract the ImageCAS dataset from the KaggleHub cache to
    `IMAGECAS_DATA_FOLDER/tmp`.

    Returns:
        None
    """
    kagglehub_cache_dir = Path(os.environ["KAGGLEHUB_CACHE"])
    full_kaggle_data_dir = kagglehub_cache_dir.joinpath("datasets",
                                                        "xiaoweixumedicalai",
                                                        "imagecas",
                                                        "versions",
                                                        "3")
    output_folder = Path(os.environ["IMAGECAS_DATA_FOLDER"]) / "tmp"

    output_folder.mkdir(parents=True, exist_ok=True)

    for zip_file in tqdm(full_kaggle_data_dir.glob('*.change2zip')):
        if (output_folder / zip_file.stem).exists():
            print("Skipping existing:", zip_file.stem)
            continue
        temp_file = zip_file.with_suffix('.zip')
        shutil.copy(zip_file, temp_file)

        # extract using 7zip:
        seven_zip_path = Path(os.environ["7ZIP_PATH"])
        subprocess.run([seven_zip_path,
                        "x",
                        str(temp_file),
                        f"-o{output_folder}"],
                        check=True)

        if temp_file.exists():
            temp_file.unlink()

def rename_imagecas_dataset():
    """
    Rename the extracted ImageCAS dataset files from `1.img.nii.gz` to
    `001.img.nii.gz` format and move them into `IMAGECAS_DATA_FOLDER/niigz`.

    Returns:
        None
    """
    input_folder = Path(os.environ["IMAGECAS_DATA_FOLDER"]) / "tmp"
    output_folder = Path(os.environ["IMAGECAS_DATA_FOLDER"]) / "niigz"
    output_folder.mkdir(parents=True, exist_ok=True)
    (input_folder / 'dont_remove_empty_folders').touch()

    for fname in input_folder.glob('*/*.nii.gz'):
        base = f'{int(fname.name.split(".")[0]):04d}'
        output_file = output_folder / (base + '.' + '.'.join(fname.name.split('.')[1:]))
        shutil.move(fname, output_file)
