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

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# Use COCO_DIR under $TMPDIR (fast scratch on HPC)
export COCO_DIR="$TMPDIR/coco"
mkdir -p "$COCO_DIR"

# Download validation images
if [ ! -f "$COCO_DIR/val2017.zip" ]; then
    echo "Downloading val2017.zip"
    wget -q http://images.cocodataset.org/zips/val2017.zip -O "$COCO_DIR/val2017.zip"
fi

# Download annotations
if [ ! -f "$COCO_DIR/annotations_trainval2017.zip" ]; then
    echo "Downloading annotations_trainval2017.zip"
    wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O "$COCO_DIR/annotations_trainval2017.zip"
fi

# Download training images
if [ ! -f "$COCO_DIR/train2017.zip" ]; then
    echo "Downloading train2017.zip"
    wget -q http://images.cocodataset.org/zips/train2017.zip -O "$COCO_DIR/train2017.zip"
fi

echo "Unzipping files into $COCO_DIR"

unzip -q "$COCO_DIR/val2017.zip" -d "$COCO_DIR"
echo "Unzipped val2017.zip to $COCO_DIR"
unzip -q "$COCO_DIR/annotations_trainval2017.zip" -d "$COCO_DIR"
echo "Unzipped annotations_trainval2017.zip to $COCO_DIR"
unzip -q "$COCO_DIR/train2017.zip" -d "$COCO_DIR"
echo "Unzipped train2017.zip to $COCO_DIR"

# Uncomment to clean up zip files and save space
# rm -f "$COCO_DIR/"*.zip

export DATA_DIR="$COCO_DIR"
export NO_ALBUMENTATIONS_UPDATE=1

export TRAIN_DIR="$DATA_DIR/train2017"
export VAL_DIR="$DATA_DIR/val2017"
export TRAIN_ANNOTATION_DIR="$DATA_DIR/annotations/instances_train2017.json"
export VAL_ANNOTATION_DIR="$DATA_DIR/annotations/instances_val2017.json"
    

echo "Running trainer.py"
python3 "$HOME/Neural-Template-Matching/temp_matching/trainer.py"
