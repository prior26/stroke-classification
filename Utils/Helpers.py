import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torchinfo import summary
import json
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd
from scipy.spatial.distance import jensenshannon


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Logger import MyLogger
from Models import TransNets, ConvNets, HybridNets
from Config import Config

def load_model(CONFIG:Config, LOGGER:MyLogger, fold:int=0, load_best=False, fineTune = False, num_classes = 3):
    torch.cuda.empty_cache()
    CONFIG.updateFold(fold)
    
    if "SWIN" in CONFIG.MODEL_NAME:
        model = TransNets.SWIN(model_size=CONFIG.MODEL_SIZE, num_classes=num_classes, freezeToLayer=CONFIG.FREEZE_TO_LAYER, pretrained = not fineTune, logger=LOGGER)
    elif "CvT" in CONFIG.MODEL_NAME:
        model = TransNets.CvT(model_size=CONFIG.MODEL_SIZE, num_classes=num_classes, freezeToLayer=CONFIG.FREEZE_TO_LAYER, pretrained = not fineTune, logger=LOGGER)
    elif "MaxViT" in CONFIG.MODEL_NAME:
        model = TransNets.MaxViT(model_size=CONFIG.MODEL_SIZE, num_classes=num_classes, freezeToLayer=CONFIG.FREEZE_TO_LAYER, pretrained = not fineTune)
    elif "ResNet" in CONFIG.MODEL_NAME:
        model = ConvNets.ResNet(model_size=CONFIG.MODEL_SIZE, num_classes=num_classes, freezeToLayer=CONFIG.FREEZE_TO_LAYER, pretrained = not fineTune, logger=LOGGER)
    elif "CustomSD" in CONFIG.MODEL_NAME:
        # model = HybridNets.CustomCT(model_size=CONFIG.MODEL_SIZE, num_classes=num_classes, device=CONFIG.DEVICE)
        model = HybridNets.StrokeDetector(model_size=CONFIG.MODEL_SIZE, num_classes=num_classes, fold=fold+1, strategy=CONFIG.STRATEGY, device=CONFIG.DEVICE, logger=LOGGER)
    else:
        LOGGER.log(f"Error: {CONFIG.MODEL_NAME} not recognized!")
        exit(1)
    
    model.to(CONFIG.DEVICE)
    optim = torch.optim.SGD(params=model.parameters(), lr=CONFIG.LEARNING_RATE, weight_decay=0.0005, dampening=0, momentum=0.9, nesterov=True)      
    if "RLRP" in CONFIG.LR_SCHEDULAR:
        lr_schedular = ReduceLROnPlateau(optim, factor=CONFIG.LRS_FACTOR, patience=CONFIG.LRS_PATIENCE)
    elif "CAWR":
        lr_schedular = CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=1)
    else:
        LOGGER.log(f"\t Invalid Schedular Key: [{CONFIG.LR_SCHEDULAR}] should be either of [RLRP, CAWR]. Using RLRP as default")
        lr_schedular = ReduceLROnPlateau(optim, factor=CONFIG.LRS_FACTOR, patience=CONFIG.LRS_PATIENCE)
    if load_best:
        # Load the best model
        checkpoint = torch.load(CONFIG.PATH_MODEL_SAVE)
        model.load_state_dict(checkpoint["model_state_dict"])
        if fineTune:
            optim.load_state_dict(checkpoint["optimizer_state_dict"])
            lr_schedular.load_state_dict(checkpoint["lr_schedular_state_dict"])
            LOGGER.log("\t" + "+"*100)
            LOGGER.log("\t"*6 + "STARTING FINE TUNING")
            LOGGER.log("\t" + "+"*100)
            LOGGER.log(f"\tMin Train Loss: [{checkpoint['train_loss']: 0.5f}] at Epoch {checkpoint['epoch']}")
            LOGGER.log(f"\tLoading Best {CONFIG.MODEL_NAME} Model for Fold: [{fold+1}/{CONFIG.K_FOLD}]")
        
            # Open all Layers
            for _, param in model.named_parameters():
                param.requires_grad = True
        
    else:
        LOGGER.log(f"\n\tNew {CONFIG.MODEL_NAME} loaded successfully")

    save_arch(CONFIG=CONFIG, model=model, fineTune=fineTune)  
    return model, optim, lr_schedular

def save_model(CONFIG:Config, model:nn.Module, optim:torch.optim.SGD, lr_schedular:CosineAnnealingWarmRestarts, epoch, train_loss):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "lr_schedular_state_dict": lr_schedular.state_dict(),
        "train_loss": train_loss,
        "epoch": epoch
    }, CONFIG.PATH_MODEL_SAVE)

def save_arch(CONFIG:Config, model:nn.Module, fineTune=False):
    # Check Architecture Folder Exists or not
    path = f"{CONFIG.PATH_MODEL_FOLDER}Architecture/"
    if not os.path.exists(path):
        os.mkdir(path)
    
    # Write Architecture
    path += f"arch{'_FT' if fineTune else ''}_{CONFIG.EXPERIMENT_NUMBER+1}.txt" 
    last_freezed_layer = CONFIG.FREEZE_TO_LAYER if not fineTune else ""
    with open(path, "w") as f:
        f.write("="*25 + "Layer Names" + "="*25 + "\n")
        for i, (name, param) in enumerate(model.named_parameters()):
            if last_freezed_layer in name and last_freezed_layer != "":
                f.write(str(i) + ": " + name + "\t\t(freezed till here)\n")
            else:
                f.write(str(i) + ": " + name + "\n")
        f.write("="*61 + "\n")
        f.write("\n\n")
        f.write(str(summary(model, (1,) + CONFIG.IMG_SIZE, depth=8 , col_names=["input_size","output_size","num_params"], verbose=0, device=CONFIG.DEVICE)))

def scheduler_step(LOGGER:MyLogger, schedular, lr, **kwargs):
    if(isinstance(schedular, CosineAnnealingWarmRestarts)): 
        schedular.step()
        if(lr > schedular.get_last_lr()[-1]):
            LOGGER.log(f"\t\t(-) Learning Rate Decreased: [{lr: 0.2e}] --> [{schedular.get_last_lr()[-1]: 0.2e}]")
            lr = schedular.get_last_lr()[-1]
        elif(lr < schedular.get_last_lr()[-1]):
            LOGGER.log(f"\t\t(+) Learning Rate Increased: [{lr: 0.2e}] --> [{schedular.get_last_lr()[-1]: 0.2e}]")
            lr = schedular.get_last_lr()[-1]

    elif(isinstance(schedular, ReduceLROnPlateau)): 
        schedular.step(kwargs["val_loss"])
        if(lr > schedular.get_last_lr()[-1]):
            LOGGER.log(f"\t\t(-) Learning Rate Decreased: [{lr: 0.2e}] --> [{schedular.get_last_lr()[-1]: 0.2e}]")
            lr = schedular.get_last_lr()[-1]
        elif(lr < schedular.get_last_lr()[-1]):
            LOGGER.log(f"\t\t(+) Learning Rate Increased: [{lr: 0.2e}] --> [{schedular.get_last_lr()[-1]: 0.2e}]")
            lr = schedular.get_last_lr()[-1]
    else: raise Exception("LR Schedular not recognized!\nType: " + type(schedular))
    return lr

def early_stop(LOGGER:MyLogger, CONFIG:Config, p_counter, training_losses):
    if(p_counter-1 >= CONFIG.PERSIST):
        LOGGER.log("\t" + f"\tValidation Loss not decreasing for {CONFIG.PERSIST}")
        if(is_decreasing_order(training_losses[-CONFIG.PERSIST:])):
            LOGGER.log("\t" + f"\tStopping Training: Overfitting Detected")
            # Break out of Training Loop
            if(CONFIG.AUTO_BREAK): 
                p_counter = 1
                return True, p_counter
        else:
            LOGGER.log("\t" + "\tTraining Loss Fluctuating")
        
        # Unsure about Overfitting, ask the user to continue
        while(True):
            if(CONFIG.AUTO_BREAK):
                flag = "n"
                break
            flag = input("\t" + "Keep Training? (y/n) : ")
            if(flag == "y" or flag == "n"):
                break
            else:
                LOGGER.log("\t" + "Wrong Input!!\n")
        
        p_counter = 1
        if(flag == "n"):    
            return True, p_counter
        else:
            return False, p_counter
    else:
        return False, p_counter

def load_checkpoint(CONFIG:Config):
    with open(CONFIG.PATH_PERFORMANCE_FOLDER + "final_performance.json", "r") as json_file:
        final_values = json.load(json_file)
    precision_values = {}
    recall_values = {}
    f1_values = {}
    for _class in CONFIG.CLASS_NAMES:
        precision_values[_class] = final_values[_class]["precision"][:CONFIG.START_FOLD]
        recall_values[_class] = final_values[_class]["recall"][:CONFIG.START_FOLD]
        f1_values[_class] = final_values[_class]["f1-score"][:CONFIG.START_FOLD]
    
    return precision_values, recall_values, f1_values

def plot_losses(fold, training_losses, validation_losses, save_path: str, logger:MyLogger):

    epochs = range(1, len(training_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_losses, label='Train Loss', linestyle='-', color='blue')
    plt.plot(epochs, validation_losses, label='Val Loss', linestyle='--', color='orange')

    # start = len(ft_training_losses) - 1 - ft_training_losses[::-1].index(-1)
    # epochs = range(start+1+1, len(ft_training_losses) + 1)
    # plt.plot(epochs, ft_training_losses[start+1:], label='FT Train Loss', marker='o', linestyle='-', color='green')
    # plt.plot(epochs, ft_validation_losses[start+1:], label='FT Val Loss', marker='x', linestyle='--', color='red')


    # Add titles and labels
    title = f"Fold: {fold+1}\n" if fold is not None else ""
    plt.title(title + 'Training and Validation Losses Over Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.yscale('log')
    
    # Set y-axis limits
    plt.ylim(0.0001, max(max(training_losses), max(validation_losses)) * 1.2)  # Slightly higher than max loss

    # Adding a grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend
    plt.legend(fontsize=12)

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory
    if logger is not None:
        logger.log(f"\tPlot saved as {save_path}")
    else:
        print(f"\tPlot saved as {save_path}")
    # ---------------------------------------------
    # Add a combined loss for all files with log axis
    # ----------------------------------------------

def is_decreasing_order(lst: list):
    for i in range(len(lst) - 1):
        if lst[i] <= lst[i + 1]:
            return False
    return True

def get_min_val_loss(path_model_save: str):
    loss_files = []
    for file_name in os.listdir(path_model_save):
        if "LOSSES" in file_name:
            loss_files.append(file_name)

    min_val_loss = float('inf')
    for loss_file in loss_files:
        try:
            val_losses = np.loadtxt(path_model_save + loss_file, delimiter=",")[1]
            min_val_loss = min(min_val_loss, np.min(val_losses))
        except Exception as e:
            print(f"Unable to process file: {loss_file}")
            print(f"Error: {e}")
    
    return min_val_loss

def get_sample_weights(dataset, indices, name, logger: MyLogger):
    if indices is not None:
        targets = torch.tensor([dataset.targets[i] for i in indices])
        # targets = dataset.targets[indices] # For segtrain
        log_string = "\t" + name 
    else:
        targets = torch.tensor(dataset.targets)
        # targets = dataset.targets # For segtrain
        log_string = name
    class_counts = torch.bincount(targets)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[targets]
    logger.log(log_string + f" Class Counts: {class_counts}, weights: {class_weights}")
    return class_weights, sample_weights

def logTime(start_time, end_time, logger:MyLogger):
    
    elapsed_time = end_time - start_time
    # Convert to HH:MM:SS format
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_formatted = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    logger.log("\t"*10 + f"  EPOCH TIME: [{time_formatted}]")

def verify_lengths(l1:list, l2:list):
    # Determine the difference in lengths
    len_diff = abs(len(l1) - len(l2))
    
    # Append 1000 to the smaller list
    if len(l1) < len(l2):
        l1.extend([1000] * len_diff)
    elif len(l2) < len(l1):
        l2.extend([1000] * len_diff)
    
    return l1, l2

def saveAsTable(path_save: str):
    with open(path_save, 'r') as f:
        data = json.load(f)
    path_save = path_save[:-5] + ".png"
    # Convert JSON data into a DataFrame
    df: pd.DataFrame
    df = pd.DataFrame(data).T  # Transpose to get categories as rows
    df = df.map(lambda x: round(x, 3) if isinstance(x, float) else x)
    df = df.iloc[:3,:3]

    # Set up a Matplotlib figure
    fig, ax = plt.subplots(figsize=(8, len(df) * 0.8))  # Adjust figure height based on rows

    # Hide axes
    ax.axis('tight')
    ax.axis('off')

    # Render the DataFrame as a table
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     rowLabels=df.index,
                     cellLoc='center',
                     loc='center')

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Adjust table scale
    plt.savefig(path_save, dpi=300, bbox_inches='tight')

def _binarizeUsingMax(t:torch.tensor):
    max_values, _ = t.max(dim=1, keepdim=True)
    return torch.where(t == max_values, torch.tensor(1.0), torch.tensor(0.0)).numpy()

def _calcPerformMetrics(y_pred, y_true, class_names, path_save):
    y_pred = _binarizeUsingMax(y_pred)
    y_true = _binarizeUsingMax(y_true)
    report = classification_report(y_true=y_true, y_pred=y_pred, target_names=class_names, output_dict=True, zero_division=0)
    with open(path_save, 'w') as f:
        json.dump(report, f, indent=4)
    saveAsTable(path_save)
    return report

def test_model(t_model: torch.nn.Module, test_loader:ImageFolder,test_class_weights, device, path_save, class_names, logger:MyLogger):
    y_trueTensor = torch.empty(0,len(class_names))
    y_predTensor = torch.empty(0,len(class_names))
    CRITERION_TEST = torch.nn.CrossEntropyLoss(weight=test_class_weights.to(device))
    with torch.no_grad():
        test_loss = 0.0
        for test_XY in test_loader:
            x = test_XY[0].to(device)
            y = test_XY[1].to(device)
            # x = test_XY["image"].to(device)
            # y = test_XY["label"].to(device)

            y_pred =  t_model(x)
            test_loss += CRITERION_TEST(y_pred, y).item()

            y_true = torch.zeros(y.shape[0],len(class_names))
            for row in range(y.shape[0]):
                y_true[row, y[row]] = 1
            y_trueTensor = torch.vstack([y_trueTensor, y_true.cpu()])
            y_predTensor = torch.vstack([y_predTensor, torch.nn.functional.softmax(y_pred, dim=1).cpu()])

    test_loss /= len(test_loader)
    y_true_labels = torch.argmax(y_trueTensor, dim=1).numpy()
    y_pred_labels = torch.argmax(y_predTensor, dim=1).numpy()
    y_pred_probs = y_predTensor.numpy()  # This is what we need for ROC curves!

    report = _calcPerformMetrics(y_pred=y_predTensor, y_true=y_trueTensor, class_names=class_names, path_save=path_save)
    logger.log(f"\tFinal Test Loss:{round(test_loss,5)}")
    return report, y_true_labels, y_pred_labels, y_pred_probs

def test_ensemble(models: list, test_loader: ImageFolder, test_class_weights, device, path_save, class_names, logger: MyLogger):
    y_trueTensor = torch.empty(0, len(class_names))
    y_predTensor = torch.empty(0, len(class_names))
    CRITERION_TEST = torch.nn.CrossEntropyLoss(weight=test_class_weights.to(device))

    with torch.no_grad():
        test_loss = 0.0
        step = 0
        for test_XY in test_loader:
            if step == 0: 
                print("Testing: |", end="\r") 
                step +=1
            elif step == 1: 
                print("Testing: /", end="\r")
                step +=1
            else:
                print("Testing: \\", end="\r")
                step = 0
            x = test_XY[0].to(device)
            y = test_XY[1].to(device)
            # x = test_XY["image"].to(device)
            # y = test_XY["label"].to(device)


            # Aggregate predictions from all models
            y_pred_ensemble = torch.zeros(x.size(0), len(class_names)).to(device)
            for model in models:
                model.eval()  # Ensure the model is in evaluation mode
                y_pred_ensemble += torch.nn.functional.softmax(model(x), dim=1)
            
            # Average predictions
            y_pred_ensemble /= len(models)

            # Calculate loss using averaged predictions
            test_loss += CRITERION_TEST(y_pred_ensemble, y).item()

            # Convert true labels to one-hot encoding
            y_true = torch.zeros(y.shape[0], len(class_names)).to(device)
            for row in range(y.shape[0]):
                y_true[row, y[row]] = 1

            # Append true and predicted tensors
            y_trueTensor = torch.vstack([y_trueTensor, y_true.cpu()])
            y_predTensor = torch.vstack([y_predTensor, y_pred_ensemble.cpu()])

    # Compute average test loss
    test_loss /= len(test_loader)

    # Calculate performance metrics
    report = _calcPerformMetrics(y_pred=y_predTensor, y_true=y_trueTensor, class_names=class_names, path_save=path_save)
    logger.log(f"\tFinal Test Loss (Ensemble): {round(test_loss, 5)}")

    return report

def normalize_img(img):
    total = np.sum(img)
    return img/total

def get_confidence_score(cam: list):
    normalized_imgs = [normalize_img(img) for img in cam]
    avg_dist = np.mean(normalized_imgs, axis=0)
    js_divergence = sum(
        1/len(normalized_imgs) * jensenshannon(img.flatten(), avg_dist.flatten(), base=2)**2 for img in normalized_imgs
    )
    return (1-js_divergence)*100

# def load_model(model_name, path):
#     if "SWIN" in model_name:
#         model = TransNets.SWIN(model_size="s")
#     elif "CvT" in model_name:
#         model = TransNets.CvT(model_size="s")
#     elif "MaxViT" in model_name:
#         model = TransNets.MaxViT(model_size="s")
#     elif "ResNet" in model_name:
#         model = ConvNets.ResNet(model_size="s")
#     else:
#         print(f"Error: {model_name} not recognized!")
#         exit(1)

#     checkpoint = torch.load(path, "cuda:0")
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.eval()
#     return model

# def make_confusion_matrix(y_true_folds:list, y_pred_folds:list, n_classes:int, class_names:list, path_save:str):
#     print("\t\t======= Making Confusion Matrix ========")
#     cms = []
#     for y_true, y_pred in zip(y_true_folds, y_pred_folds):
#         cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
#         cms.append(cm)

#     cms = np.array(cms)  # shape: (10, n_classes, n_classes)
#     cm_mean = np.mean(cms, axis=0)
#     cm_std = np.std(cms, axis=0)
#     plt.figure(figsize=(8, 6))
#     annot = np.empty_like(cm_mean).astype(str)

#     for i in range(n_classes):
#         for j in range(n_classes):
#             annot[i, j] = f'{cm_mean[i, j]:.1f}±{cm_std[i, j]:.1f}'

#     sns.heatmap(cm_mean, annot=annot, fmt='', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title('Confusion Matrix (Mean ± Std over 10 folds)')
#     # plt.show()
#     plt.savefig(path_save, dpi=300, bbox_inches='tight')
#     print("\t\t======= DONE ========\n")

# Averaged confusion matrix function (your original approach - shows reliability)
def make_confusion_matrix(y_true_folds: list, y_pred_folds: list, n_classes: int, class_names: list, path_save: str):
    """
    Create confusion matrix by averaging across CV folds to show model reliability.
    
    Args:
        y_true_folds: List of true labels for each fold
        y_pred_folds: List of predicted labels for each fold  
        n_classes: Number of classes
        class_names: List of class names for labels
        path_save: Path to save the confusion matrix plot
    """
    print("\t\t======= Making Confusion Matrix (Averaged) ========")
    # Set global font sizes for all plots
    plt.rcParams.update({
        'font.size': 14+2,          # General font size
        'axes.titlesize': 16+2,     # Title font size
        'axes.labelsize': 14+2,     # X and Y label font size
        'xtick.labelsize': 12+2,    # X tick label font size
        'ytick.labelsize': 12+2,    # Y tick label font size
        'legend.fontsize': 12+2,    # Legend font size
        'figure.titlesize': 18+2    # Figure title (suptitle) font size
    })
    
    # Create confusion matrices for each fold
    cms = []
    for y_true, y_pred in zip(y_true_folds, y_pred_folds):
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
        cms.append(cm)
    
    cms = np.array(cms)  # shape: (n_folds, n_classes, n_classes)
    cm_mean = np.mean(cms, axis=0)
    cm_std = np.std(cms, axis=0)
    
    # Create visualization
    plt.figure(figsize=(8, 6))
    
    # Create annotations with mean ± std
    annot = np.empty_like(cm_mean).astype(str)
    for i in range(n_classes):
        for j in range(n_classes):
            annot[i, j] = f'{cm_mean[i, j]:.1f}±{cm_std[i, j]:.1f}'
    
    sns.heatmap(cm_mean, annot=annot, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # plt.title('Confusion Matrix (Mean ± Std over CV folds)')
    # plt.suptitle(f'Model Reliability Analysis - {len(y_true_folds)} folds', 
    #              y=1.02, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(path_save, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    
    print(f"\t\tConfusion matrices averaged across {len(y_true_folds)} folds")
    print(f"\t\tMean diagonal elements: {np.diag(cm_mean)}")
    print(f"\t\tStd diagonal elements: {np.diag(cm_std)}")
    print("\t\t======= DONE ========\n")
    plt.rcParams.update(plt.rcParamsDefault)
    return cm_mean, cm_std  # Return both mean and std for further analysis


# ROC curve function (concatenated approach - for overall performance)
def make_roc_curve_macro(y_true_folds: list, y_prob_folds: list, n_classes: int, 
                        class_names: list, path_save: str):
    """
    Create macro-average ROC curve by concatenating predictions from all CV folds.
    This shows overall classification performance across all data.
    
    Args:
        y_true_folds: List of true labels for each fold (integers)
        y_prob_folds: List of predicted probabilities for each fold (shape: n_samples x n_classes)
        n_classes: Number of classes
        class_names: List of class names for labels
        path_save: Path to save the ROC curve plot
    """
    plt.rcParams.update({
        'font.size': 14+2,          # General font size
        'axes.titlesize': 16+2,     # Title font size
        'axes.labelsize': 14+2,     # X and Y label font size
        'xtick.labelsize': 12+2,    # X tick label font size
        'ytick.labelsize': 12+2,    # Y tick label font size
        'legend.fontsize': 12+2,    # Legend font size
        'figure.titlesize': 18+2    # Figure title (suptitle) font size
    })
    print("\t\t======= Making ROC Curve (Macro-Average) ========")
    
    # Concatenate all fold results
    y_true_all = np.concatenate(y_true_folds)
    y_prob_all = np.concatenate(y_prob_folds, axis=0)
    
    total_samples = len(y_true_all)
    print(f"\t\tTotal samples processed: {total_samples}")
    
    # Binarize the labels for multiclass ROC
    y_true_bin = label_binarize(y_true_all, classes=np.arange(n_classes))
    
    # Calculate ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(8, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob_all[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        # plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
        #         label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    # Calculate micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob_all.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle='--', lw=2,
            label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')
    
    # Calculate macro-average ROC curve
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    
    # plt.plot(all_fpr, mean_tpr, color='navy', linestyle='--', lw=3,
    #         label=f'Macro-average (AUC = {macro_auc:.3f})')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1, alpha=0.8)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve with Micro Average')
    # plt.suptitle(f'Overall Classification Performance - {total_samples} total samples from {len(y_true_folds)} folds', 
    #              y=0.98, fontsize=10)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path_save, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\t\tMacro-average AUC: {macro_auc:.3f}")
    print(f"\t\tIndividual class AUCs: {[f'{class_names[i]}: {roc_auc[i]:.3f}' for i in range(n_classes)]}")
    print("\t\t======= DONE ========\n")
    plt.rcParams.update(plt.rcParamsDefault)
    return macro_auc, roc_auc


if __name__ == "__main__":
    import cv2
    img1 = cv2.imread('CAM_0.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img2 = cv2.imread('CAM_1.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)

    print(get_confidence_score([img1, img2]))
