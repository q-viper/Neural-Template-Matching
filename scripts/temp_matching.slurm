#!/bin/bash -l
#
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=temp_job.out

unset SLURM_EXPORT_ENV

module load python
conda activate pytorch

# pip --proxy http://proxy:80 install --user segmentation_models_pytorch==0.3.3 scikit-learn
# download if the data is not already downloaded
# export proxy
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
mkdir -p ./coco
if [ ! -f ./coco/val2017.zip ]; then
    echo "Downloading val2017.zip"
    # download but hide progress bar
    wget -q http://images.cocodataset.org/zips/val2017.zip -O ./coco/val2017.zip
fi

if [ ! -f ./coco/annotations_trainval2017.zip ]; then
    echo "Downloading annotations_trainval2017.zip"
    wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O ./coco/annotations_trainval2017.zip
fi

if [ ! -f ./coco/train2017.zip ]; then
    echo "Downloading train2017.zip"
    wget -q http://images.cocodataset.org/zips/train2017.zip -O ./coco/train2017.zip
fi

echo "Unzipping files to $TMPDIR/coco"
mkdir -p "$TMPDIR/coco"
unzip -q "$HOME/temp_matching/coco/val2017.zip" -d "$TMPDIR/coco"
echo "Unzipped val2017.zip"
unzip -q "$HOME/temp_matching/coco/annotations_trainval2017.zip" -d "$TMPDIR/coco"
echo "Unzipped annotations_trainval2017.zip"
unzip -q "$HOME/temp_matching/coco/train2017.zip" -d "$TMPDIR/coco"
echo "Unzipped train2017.zip"

# delete downloaded files
# rm -rf ./coco

export DATA_DIR="$TMPDIR/coco"
export NO_ALBUMENTATIONS_UPDATE=1

echo "Running trainer.py"

python3 /home/hpc/rlvl/rlvl102h/temp_matching/trainer.py