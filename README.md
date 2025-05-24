# Towards Reliable and Interpretable Predictions in Stroke Detection Using Computed Tomography Scans
This repository contains the source code accompanying the research paper titled _"Towards Reliable and Interpretable Predictions in Stroke Detection Using Computed Tomography Scans"_ submitted to Eighth AAAI/ACM Conference on AI, Ethics, and Society, **(AIES)**. The code is provided to support reproducibility and further research.
This code is part of a **double-blind** peer review submission. **All identifying details have been anonymized** to comply with AIES guidelines.

## Setup
### Dataset
The steps to set up the data for training and testing are -
1. Open [this link](https://acikveri.saglik.gov.tr/Home/DataSetDetail/1) in the browser. 
2. For each file, click on `Detail` and download all the compressed files.
3. After extracting all the compressed files, copy the images from 
    - `Kanama Veri Seti/PNG/` to `Data/Compiled/Train/Hemorrhagic/`
    - `Iskemik Veri Seti/PNG/` to `Data/Compiled/Train/Ischemic/`
    - `Kontur Yok Diger Veri Seti/` to `Data/Compiled/Train/Normal/`

    This data has the problem of imbalance - 

    | **Class**       | **Total Train** | **Train Percent** | **Total Test** | **Test Percent** |
    |---------------|--------------|----------------|-------------|---------------|
    | **Hemorrhagic** | 874          | 16.4%          | 219         | 16.5%         |
    | **Ischemic**    | 904          | 17.0%          | 226         | 17.0%         |
    | **Normal**      | 3542         | 66.6%          | 885         | 66.5%         |


4. Initially inside each folder in `Data/Compiled` there is a `temp.txt` file that serves as a placeholder for keeping the directory structure intact. Make sure you delete these `temp.txt` files before running any python scripts.

### Environment Setup
For setting up the environment, **Anaconda** or **Miniconda** must be installed. 
Navigate through the terminal to this project folder and just run - 

```bash
conda env create -p ./env -f environment.yml
```

Once the environment has been created just activate the environment by running -
```bash
conda activate ./env
``` 

## Training
There are two types of classifiers used in this project.
- Convolutional Networks
    - Residual Network (ResNet)
- Transformer Networks
    - Shifted Window (SWIN) Transformer
    - Convolutional Vision Transformer (CvT) 

The directories are arranged according to this classification. Each model-folder has 2 sub-folders
1. `K-Fold`: For all results regarding K-Fold cross validation for multi-class classification.
    
    - After training the model using K-Fold training paradigm (multi-class classification), the final result of the model are written as -
        1. `K-Fold/final_result.json`: Contains the final precision, recall and f1-score values.
        2. `K-Fold/Performance/confusion_matrix.png`: Contains the k-fold averaged confusion matrix.
        3. `K-Fold/Performance/roc_curve.png`: Contains the micro-averaged ROC curve.

2. `Simple`: For all results regarding training each model for binary classification.
    - After training the model using Simple training paradigm (binary classification), the final performance can be found in `Simple/Performance/final_performance.json` and the training loss curve can be found in `Simple/Plots/lossplot.png`

To initialize the training process there are two `init.json` files that are used. 
1. **Model Specific**: The `init.json` file inside the model-folder. <br> It contains the model specific parameters to be used while training that model.

    | **Parameter**         | **Description**                                                                 |
    |-----------------------|---------------------------------------------------------------------------------|
    | `"BATCH_SIZE"`       | The batch size to be used while training this model.                            |
    | `"BATCH_LOAD"`      | As gradient accumulation is being used, this parameter is used for low resource machines. Always, BATCH_LOAD <= BATCH_SIZE. |
    | `"LEARNING_RATE"`    | The learning rate used for training this model.                                |
    | `"COMPLETED_FOLD"`   | While training with K-Fold cross-validation, this will track the number of folds that have been successfully completed. This helps if for some reason the machine crashes and you have to restart the training from the last completed fold. It is automatically updated while training. |
    | `"LR_SCHEDULAR"`     | This is the scheduler that has to be used in the freezed-training stage. The two options are: - _`"CAWR"`_: Cosine Annealing Warm Restarts - _`"RLRP"`_: Reduce Learning Rate on Plateau |
    | `"LRS_PATIENCE"`     | This is the patience parameter that will be used by the _"RLRP"_ scheduler. This will be used in the fine-tuning stage. It will also be used in the freezed-training stage if _"RLRP"_ is selected as the scheduler in the previous argument. |
    | `"LRS_FACTOR"`       | The factor by which the learning rate decreases for _"RLRP"_ scheduler.         |
    | `"FREEZE_TO_LAYER"`  | The layer of the model up to and including which all learnable parameters will be frozen. |



2. **Training Specific**: Tne `init.json` file in the project folder. <br> It contains the training specific parameters to be used while training any model.

    | **Key**              | **Description** |
    |----------------------|---------------|
    | MODEL_NAME       | Name of the model <br>Accepted Values: `["ResNet_S", "SWIN_S", "CvT_S"]`|
    | MODEL_CUSTOM_NAME| Custom Name of the model, used for folder creation (Optional) <br>Accepted Values: `str`|
    | MODEL_TYPE       | Type of model <br>Accepted Values: `["conv", "trans", "hybrid"]` |
    | DEVICE           | Specifies the computing device <br>Accepted Values: `["cuda:{num}", "cpu"]`. |
    | STRATEGY         | Specifies the strategy of ensembling if hybrid is is used as model type<br>Accepted Values: `["FCBW","QCA", "QSA"]` |
    | LOAD_CHECKPOINT  | Restarts K-fold training from the last completed checkpoint<br>Accepted Values: `[true, false]`. |
    | K_FOLD           | Number of folds. <br> Accepted Values: `[integer]`; `>0` for k-fold, `<0` for simple training |
    | TRAIN_EPOCHS     | Epochs for the freezed-training phase. <br> Accepted Values: `[+int]`|
    | FINE_TUNE_EPOCHS | Epochs for the fine-tuning phase. <br> Accepted Values: `[+int]`|
    | PERSISTENCE      | Early Stopping persistence <br> Accepted Values: `[+int]` |
    | IMG_SIZE         | Input image dimensions <br> Default: `[1, 224, 224]`. |
    | AUTO_BREAK       | For Early stopping, toggles auto-stop or user prompted decision <br>Acceted Values: `[true, false]` |
    | SERVER_USERNAME  | For logging purpose: Not Important. |
    | SERVER_FOLDER    | For logging purpose: Not Important. |
    | SERVER_URL       | For logging purpose: Not Important. |
    | PATH_DATASET     | Local path to the dataset <br>Default: `"Data/Compiled/"`. |

After completing all the steps from the setup and setting the values in both `init.json` files according to the requirement, just run 
```bash
python train.py
```
and the training of the specified model will begin and after training the model will also go through the testing phase. Keep in mind, that to train an ensemble all the ResNet, SWIN and CvT should already be trained and exist at the path `Classifiers/{model_type}/{model_name}/K-Fold/F{fold_number}_Checkpoint.pth`.

## Testing on Inference Test Images
There are 4 Images provided as inference images which have not been seen by any models. 
Each inference image initally contains 3 files - 
1. `img.png`: the image to run inference on.
2. `overlay.png`: the hand annotation of the inference image overlayed on the image itself.
3. `result.json`: contains the actual class of the image `class`, the predicted class of the image `pred`, and Jenson-Shanon Score for this image `JSC`.    
    For any inference image, the values of keys `pred` and `JSC` gets updated after the the script `infer.py` is run. 


To run inference on these images just run - 
```bash
python infer.py
```
this will run inference on the images using the ensemble model trained on the 1st fold for multi-class classification. 
Optionally you can also provide the fold you want to use by instead using - 
```bash
python infer.py --fold {fold_number}
```
You should replace `{fold_number}` with the fold number you want to use.

Running `infer.py` creates:

- An image **`final_output.png`** inside `Inference Images/Image {num}/{model_name}/`. Example:  
  <p align="center"><img src="Inference Images/Image 1/CustomSD_S_QSA/final_output.png" width="70%" height="auto"/></p>

- Masks generate by each of the models in the **`masks/`** folder inside `Inference Images/Image {num}/{model_name}/`. Example:

  | <p align="center">**ResNet-S.png**</p> | <p align="center">**SWIN-S.png**</p> | <p align="center">**CvT-S.png**</p> |
  |-------------|------------|------------|
  | <p align="center"><img src="Inference Images/Image 1/CustomSD_S_QSA/masks/resnet.png" width="100%" height="auto"/></p> | <p align="center"><img src="Inference Images/Image 1/CustomSD_S_QSA/masks/swin.png" width="100%" height="auto"/></p> | <p align="center"><img src="Inference Images/Image 1/CustomSD_S_QSA/masks/cvt.png" width="100%" height="auto"/></p> |


