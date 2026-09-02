"""
Script to download and prepare the CAISA related datasets.
"""

from data_download import (
    download_imagecas_dataset,
    extract_imagecas_dataset,
    rename_imagecas_dataset,
)

if __name__ == '__main__':
    print('Starting download and preparation of the ImageCAS dataset...')

    download_imagecas_dataset()
    extract_imagecas_dataset()
    rename_imagecas_dataset()

    print('Done downloading and preparing the ImageCAS dataset.')
