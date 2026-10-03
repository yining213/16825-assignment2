import os

# root location where the dataset was downloaded: this folder by default, or set L3D_DATA_ROOT
root_location = os.environ.get("L3D_DATA_ROOT", os.path.dirname(os.path.abspath(__file__)))
# set L3D_FULL_DATASET=1 to use the three-class (chair, car, plane) dataset
use_full_dataset = os.environ.get("L3D_FULL_DATASET", "0") == "1"
dataset_name = (
    "r2n2_shapenet_dataset_full" if use_full_dataset else "r2n2_shapenet_dataset"
)

R2N2_PATH = f"{root_location}/{dataset_name}/r2n2"
SHAPENET_PATH = f"{root_location}/{dataset_name}/shapenet"

if use_full_dataset:
    SPLITS_PATH = f"{root_location}/{dataset_name}/split_3c.json"  # split file contains data entry for 3 classes
else:
    SPLITS_PATH = f"{root_location}/{dataset_name}/split_03001627.json"  # split file contains data entry for 03001627 class
