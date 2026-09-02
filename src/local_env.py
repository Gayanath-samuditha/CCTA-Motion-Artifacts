import os
from pathlib import Path


def set_environment_variables():
    data_dir = Path(__file__).resolve().parent.parent / 'data'

    environ_dict = {"CAISA_DATA_FOLDER": data_dir,
                    "IMAGECAS_DATA_FOLDER": data_dir / "datasets" / "imagecas",
                    "KAGGLEHUB_CACHE": data_dir / 'datasets' / 'kaggle_cache',
                    "7ZIP_PATH": Path("C:/Program Files/7-Zip/7z.exe")}

    # Convert all values to string and update envs
    environ_dict = {k: str(v) for k, v in environ_dict.items()}
    os.environ.update(environ_dict)