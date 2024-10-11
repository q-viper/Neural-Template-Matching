# Template Matching Using Deep Learning
An experiment to do template matching based on neural networks.

## Training Procedure
* Prepare a `.venv` file that contains following:
```env
TRAIN_DIR=assets\training_data\train2017\train2017
TRAIN_ANNOTATION_DIR=assets\training_data\annotations_trainval2017\annotations\instances_train2017.json
VAL_DIR=assets\training_data\val2017\val2017
VAL_ANNOTATION_DIR=assets\training_data\annotations_trainval2017\annotations\instances_val2017.json
```
Had to do this to make it compatible with HPC. Slurm job is in [scripts](/scripts).

## Results
### Experiment: `2024-09-24`
* Encoder: `ResNet152`
* Train Data Per Epoch: 10000
* Valid Data Per Epoch: 500
* Optimizer: Adam with Lr=0.0001
* Loss function: DiceLoss
* Training Curve
![](assets/2024-09-24/loss_iou.png)
* Predictions at [assets/2024-09-24/]
![](assets/2024-09-24/epoch_250.png)


## Inference
A [live_run.py](/live_run.py) should work out of the box. First compute the encodings of query and search based on that.

## References
