"""
This script converts the ImageCAS dataset structure to the nnUNet format
Reads ImageCAS data from niigz/ folder and organize it according to the nnU-net v2 specifications with the naming conventions
https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md
"""

import os
import json 
import shutil
from pathlib import Path

from local_env import set_environment_variables
set_environment_variables()


def get_imagecas_files():
    """
    Get sorted lists of ImageCAS images and labels.
    """
    source_dir=Path(os.environ["IMAGECAS_DATA_FOLDER"])/"niigz"

    images = sorted(source_dir.glob("*.img.nii.gz"))
    labels = sorted(source_dir.glob("*.label.nii.gz"))

    if len(images) !=len(labels):
        raise ValueError(f"File count mismatch: {len(images)} images but {len(labels)} labels")
    
    print(f"Found {len(images)} image-label pairs ")
    return images,labels



def setup_nnunet_folders():
    #Create nnUnet directory structure
    base_dir = Path(os.environ["CAISA_DATA_FOLDER"])/"nnUNet_raw"
    dataset_dir = base_dir / "Dataset001_ImageCAS"

    for folder in ["imagesTr","labelsTr", "imagesTs"]:
        (dataset_dir/folder).mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory structure at {dataset_dir}")
    return dataset_dir



def split_dataset(images, labels, train_fraction=0.8):
    #Splitting data into training and testing sets.
    split_idx = int(len(images)*train_fraction)

    train_imgs = images[:split_idx]
    test_imgs = images[split_idx:]
    train_lbls = labels[:split_idx]

    print(f"split:{len(train_imgs)} training images, {len(test_imgs)} testing images")
    return train_imgs, train_lbls, test_imgs


def copy_training_data(images, labels, dataset_dir):
    #Copy training images and labels with nnUnet convention

    for img, lbl in zip(images, labels):
        case_id=img.name.split('.')[0]


        #Training images-> ImageCAS_0001_0000.nii.gz
        img_dest= dataset_dir/"imagesTr"/f"ImageCAS_{case_id}_0000.nii.gz"
        shutil.copy(img,img_dest)

        #Training Labels
        lbl_dest=dataset_dir/"labelsTr"/f"ImageCAS_{case_id}.nii.gz"
        shutil.copy(lbl, lbl_dest)


def copy_test_data(images, dataset_dir):
    #Copy test images with nnU-Net naming convention.

    for img in images:
        case_id = img.name.split('.')[0]
        img_dest = dataset_dir / "imagesTs" / f"ImageCAS_{case_id}_0000.nii.gz"
        shutil.copy(img, img_dest)


def create_dataset_json(dataset_dir, n_training, n_test):
    #Generate dataset.json with required metadata.

    metadata = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "coronary_artery": 1},
        "numTraining": n_training,
        "numTest": n_test,
        "file_ending": ".nii.gz"
    }
    
    json_path = dataset_dir / "dataset.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created dataset.json")



def verify_output(dataset_dir):
    #verify conversion  completed successfully
    
    train_imgs = list((dataset_dir/ "imagesTr").glob("*.nii.gz"))
    train_lbls = list((dataset_dir/ "labelsTr").glob("*.nii.gz"))
    test_imgs = list((dataset_dir / "imagesTs").glob("*.nii.gz"))

    print(f"\nverification:")
    print(f"Trainng images= {len(train_imgs)}")
    print(f"Training labels={len(train_lbls)}")
    print(f"Test images={len(test_imgs)}")

    if len(train_imgs) != len(train_lbls):
        print(f"Training Data Mismathc!")
    else:
        print(f"Training data Ok!")


def main():
    #convert ImageCAS to nnU-net format

    print("converting ImageCAS to nnU-net format")

    images, labels = get_imagecas_files()
    dataset_dir= setup_nnunet_folders()

    train_imgs, train_lbls, test_imgs = split_dataset(images, labels)

    copy_training_data(train_imgs, train_lbls, dataset_dir)

    copy_test_data(test_imgs, dataset_dir)

    create_dataset_json(dataset_dir, len(train_imgs), len(test_imgs))

    verify_output(dataset_dir)

    print(f"\nConversion complete. Dataset ready at:\n{dataset_dir}")


if __name__=='__main__':
    main()    