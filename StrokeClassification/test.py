import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from Utils.Helpers import *
from Config import Config
from Models import ConvNets, TransNets
from Logger import MyLogger
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
import numpy as np
import os
from sklearn.model_selection import KFold
from torchinfo import summary
import time

def test(multiclass: bool):
    global MODELS
    if not os.path.exists("Results/"):
        os.mkdir("Results/")

    LOGGER.log("\t" + "-"*100)
    if multiclass:
        LOGGER.log("\t" + f"\t\tFOLD[{FOLD}] Testing Ensemble")
        LOGGER.log("\t" + f"Hemorrhagic Vs Ischemic Vs Normal")
        path_save_1 = "Results/Multiclass Report.json"
        path_save_2 = "Results/Multiclass Performance.json"
    else:
        LOGGER.log("\t" + f"\t\tTesting Ensemble")
        LOGGER.log("\t" + f"Normal Vs Stroke")
        path_save_1 = "Results/BinaryClass Report.json"
        path_save_2 = "Results/BinaryClass Performance.json"
    LOGGER.log("\t" + "-"*100)
    # Performance Metrics
    precision_values = {key: [] for key in CONFIG.CLASS_NAMES}
    recall_values = {key: [] for key in CONFIG.CLASS_NAMES}
    f1_values = {key: [] for key in CONFIG.CLASS_NAMES}

    # Create Test loader
    test_class_weights, test_sample_weights = get_sample_weights(CONFIG.TEST_DATA.dataset, CONFIG.TEST_DATA.indices, "Test", logger = LOGGER)
    test_loader = DataLoader(dataset = CONFIG.TEST_DATA, batch_size = CONFIG.BATCH_SIZE, num_workers=CONFIG.WORKERS//2, pin_memory=True, generator=CONFIG.GENERATOR, persistent_workers=True)

    report = test_ensemble(
        models=MODELS,
        test_loader = test_loader,
        test_class_weights = test_class_weights,
        device = CONFIG.DEVICE,
        path_save=path_save_1,
        class_names=CONFIG.CLASS_NAMES,
        logger = LOGGER
    )

    for _class in CONFIG.CLASS_NAMES:
        metrics = report[_class]
        precision_values[_class].append(metrics["precision"])
        recall_values[_class].append(metrics["recall"])
        f1_values[_class].append(metrics["f1-score"])
        LOGGER.log(f"\n\tClass: {_class}")
        LOGGER.log(f"\t\t|--- Precision: {metrics['precision']: 0.5f}")
        LOGGER.log(f"\t\t|--- Recall: {metrics['recall']: 0.5f}")
        LOGGER.log(f"\t\t|--- F1-Score: {metrics['f1-score']: 0.5f}")
    
    with open(path_save_2, "w") as json_file:
        json.dump(report, json_file, indent=4)
    
    for model in MODELS:
        del model
    MODELS = []
    LOGGER.log("\t" + "-"*100)

def load_model(model_config:tuple, multiclass:bool):
    model_name = model_config[0]
    model_type = model_config[1]
    fold = model_config[2]
    if not multiclass:
        path = f"Classifiers/{MODEL_TYPE_DICT[model_type]}/{model_name}/Simple/Checkpoint.pth"
        num_classes = 2
    else:
        # path = f"Classifiers/{MODEL_TYPE_DICT[model_type]}/{model_name}/K-Fold/F{fold}_Checkpoint.pth"
        path = f"Classifiers/{MODEL_TYPE_DICT[model_type]}/{model_name}/Simple/Checkpoint.pth"
        num_classes = 3

    if "SWIN" in model_name:
        model = TransNets.SWIN(model_size="s", num_classes=num_classes)
    elif "CvT" in model_name:
        model = TransNets.CvT(model_size="s", num_classes=num_classes)
    elif "MaxViT" in model_name:
        model = TransNets.MaxViT(model_size="s", num_classes=num_classes)
    elif "ResNet" in model_name:
        model = ConvNets.ResNet(model_size="s", num_classes=num_classes)
    else:
        print(f"Error: {model_name} not recognized!")
        exit(1)

    checkpoint = torch.load(f=path, map_location=CONFIG.DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(CONFIG.DEVICE)
    return model

if __name__ == "__main__":
    CONFIG = Config()
    MODEL_TYPE_DICT: dict = {"conv": "Convolutional Networks", "trans" : "Transformer Networks"}
    LOGGER = MyLogger(
        server_url = CONFIG.SERVER_URL,
        server_username = CONFIG.SERVER_USERNAME,
        server_folder = CONFIG.SERVER_FOLDER,
        model_name = CONFIG.MODEL_NAME,
        path_localFile = "Results/results_log.txt"
    )
    FOLD = 1
    MODEL_CONFIGS = [
        ("ResNet_S_new", "conv", FOLD),
        ("SWIN_S_new", "trans", FOLD),
        ("CvT_S_new", "trans", FOLD)
    ]
    CONFIG.BATCH_SIZE = 128
    # ---------------- Test Binary Classification -----------------
    # Train test split
#    CONFIG.DATA = ImageFolder(CONFIG.PATH_DATASET_MERGE_TRAIN, CONFIG.TRANSFORMS_TRAIN)
#    CONFIG.CLASS_NAMES = CONFIG.DATA.classes
#    CONFIG.TRAIN_DATA, CONFIG.VAL_DATA, CONFIG.TEST_DATA = random_split(dataset=CONFIG.DATA, lengths=[0.7, 0.1, 0.2], generator=CONFIG.GENERATOR)
#    CONFIG.VAL_DATA.transform = CONFIG.TRANSFORMS_TEST
#    CONFIG.TEST_DATA.transform = CONFIG.TRANSFORMS_TEST
#    MODELS = []
#    for model_config in MODEL_CONFIGS:
#        MODELS.append(load_model(model_config=model_config, multiclass=False))
#    test(multiclass=False)

    # ------------------------------------------------------------
    # ---------------- Test MultiClass Classification -----------------
    MODELS = []
    for model_config in MODEL_CONFIGS:
        MODELS.append(load_model(model_config=model_config, multiclass=True))
    
    CONFIG.TRAIN_DATA, CONFIG.VAL_DATA, CONFIG.TEST_DATA = random_split(dataset=CONFIG.DATA, lengths=[0.7, 0.1, 0.2], generator=CONFIG.GENERATOR)
    CONFIG.VAL_DATA.transform = CONFIG.TRANSFORMS_TEST
    CONFIG.TEST_DATA.transform = CONFIG.TRANSFORMS_TEST
    CONFIG.CLASS_NAMES = CONFIG.DATA.classes
    # KF = KFold(n_splits=CONFIG.K_FOLD, shuffle=True, random_state=CONFIG.RANDOM_STATE)
    # for fold, (train_idx, test_idx) in enumerate(KF.split(CONFIG.TRAIN_DATA)):
    #     if fold != FOLD-1:
    #         continue
    #     else:
    #         CONFIG.TEST_DATA = Subset(CONFIG.DATA, test_idx)
    #         CONFIG.TEST_DATA.transform = CONFIG.TRANSFORMS_TEST
    #         break
    test(multiclass=True)
        

    LOGGER.log("\n\n" + "="*55 + " END " + "="*55 + "\n\n")
    
