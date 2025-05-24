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

def train_KCV():
    LOGGER.log("\n" + "#"*115 + "\n")
    LOGGER.log("\t\t\t\t\tTraining: " + CONFIG.MODEL_NAME)
    LOGGER.log("\n" + "#"*115)

    # fold_min_val_loss = []
    if CONFIG.LOAD_CHECKPOINT and CONFIG.START_FOLD > 1:
        precision_values, recall_values, f1_values = load_checkpoint(CONFIG=CONFIG)
    else:
        precision_values = {key: [] for key in CONFIG.CLASS_NAMES}
        recall_values = {key: [] for key in CONFIG.CLASS_NAMES}
        f1_values = {key: [] for key in CONFIG.CLASS_NAMES}
    try:
        for fold, (train_idx, val_idx) in enumerate(KF.split(CONFIG.TRAIN_DATA)):
            LOGGER.log("\t" + "="*100)
            LOGGER.log(f"\tFold {fold+1}/{CONFIG.K_FOLD}")
            LOGGER.log("\t" + "="*100)
            if CONFIG.LOAD_CHECKPOINT and fold+1 < CONFIG.START_FOLD:
                LOGGER.log(f"\tSkipping till: {CONFIG.START_FOLD}")
                continue 

            _train = Subset(CONFIG.TRAIN_DATA, train_idx)
            _val = Subset(CONFIG.TRAIN_DATA, val_idx)

            _, sample_weights_train = get_sample_weights(CONFIG.TRAIN_DATA, train_idx, "Train", logger = LOGGER)
            val_class_weights, sample_weights_val = get_sample_weights(CONFIG.TRAIN_DATA, val_idx, "Val", logger = LOGGER)

            
            CONFIG.CRITERION_VAL = nn.CrossEntropyLoss(weight=val_class_weights.to(CONFIG.DEVICE))
            SAMPLER_TRAIN = WeightedRandomSampler(weights=sample_weights_train , num_samples=len(sample_weights_train), replacement=True, generator=CONFIG.GENERATOR)
            
            train_loader = DataLoader(dataset = _train, batch_size = CONFIG.BATCH_LOAD, num_workers=CONFIG.WORKERS//2, pin_memory=True, sampler=SAMPLER_TRAIN, generator=CONFIG.GENERATOR, persistent_workers=True, prefetch_factor=4)
            val_loader = DataLoader(dataset = _val, batch_size = CONFIG.BATCH_LOAD, num_workers=CONFIG.WORKERS//2, pin_memory=True, generator=CONFIG.GENERATOR, persistent_workers=True, prefetch_factor=4)
            
            #Initialize New Model for current fold 
            MODEL, OPTIMIZER, LR_SCHEDULER = load_model(CONFIG=CONFIG, LOGGER=LOGGER, fold=fold)
            training_losses = []
            validation_losses = []
            p_counter = 1
            # min_val_loss = float('inf')
            min_train_loss = float('inf')
            lr = CONFIG.LEARNING_RATE

            epoch = 0
            fine_tuning = False
            total_epochs = CONFIG.TRAIN_EPOCHS + CONFIG.FINE_TUNE_EPOCHS
            for epoch in range(total_epochs):
                start_time = time.time()
                LOGGER.log("\t" + f"{('--' if not fine_tuning else '++')}" + "-"*98)
                LOGGER.log("\t" + f"FOLD: [{fold+1}/{CONFIG.K_FOLD}]")
                LOGGER.log("\t" + f"EPOCH: [{epoch+1}/{total_epochs}]" + "\t"*8 + f"PERSISTENCE: [{p_counter}/{CONFIG.PERSIST}]")
                LOGGER.log("\t" + f"{('--' if not fine_tuning else '++')}" + "-"*98)

                train_loss = 0.0
                accum_loss = 0.0
                count = 0
                # Training 1 Epoch
                MODEL.train()
                for step, train_XY in enumerate(train_loader, 0):
                    
                    # Extract X and Y
                    imgs = train_XY[0].to(CONFIG.DEVICE)
                    labels = train_XY[1].to(CONFIG.DEVICE)
                    # imgs = train_XY["image"].to(CONFIG.DEVICE)
                    # labels = train_XY["label"].to(CONFIG.DEVICE)
                    
                    # Predict labels 
                    y_pred = MODEL(imgs)

                    # Calculate Error
                    error = CONFIG.CRITERION_TRAIN(y_pred, labels)
                    error.backward()
                    accum_loss += error.item()
                    
                    print("\t" +"\tSTEP: [%d/%d]" % (step+1,len(train_loader)), end= "\r")
                    if(count*CONFIG.BATCH_LOAD >= CONFIG.BATCH_SIZE):    
                        OPTIMIZER.step()
                        OPTIMIZER.zero_grad()
                        train_loss += accum_loss
                        print("\t" +"\tSTEP: [%d/%d]\t\t\t\t\t\t>>>>>Batch Loss: [%0.5f]" % (step+1,len(train_loader),accum_loss/(count)), end = "\r") # Print avg batch loss instead of total accum loss
                        accum_loss = 0.0
                        count = 0
                    count += 1
                # avg epoch loss
                train_loss /= len(train_loader)
                training_losses.append(train_loss)
                LOGGER.log("\n\n\t" +"\tTraining Loss: [%0.5f]" % (training_losses[-1]))

                # Validation
                MODEL.eval()
                with torch.no_grad():
                    val_loss = 0
                    for val_XY in val_loader:
                        imgs = val_XY[0].to(CONFIG.DEVICE)
                        labels = val_XY[1].to(CONFIG.DEVICE)
                        # imgs = val_XY["image"].to(CONFIG.DEVICE)
                        # labels = val_XY["label"].to(CONFIG.DEVICE)

                        y_pred = MODEL(imgs)
                        val_loss += CONFIG.CRITERION_VAL(y_pred, labels).item()
                val_loss /= len(val_loader)
                validation_losses.append(val_loss)
                LOGGER.log("\t" +"\tWeighted Val Loss: [%0.5f]" % (val_loss))

                # Save Best Model with minimum training loss
                p_counter += 1
                if(train_loss < min_train_loss):
                    min_train_loss = train_loss
                    save_model(CONFIG=CONFIG, model=MODEL, optim=OPTIMIZER, lr_schedular=LR_SCHEDULER, epoch=epoch, train_loss=train_loss)
                    p_counter = 1
                LOGGER.log("\t" +"\tMinimum Training Loss: [%0.5f]" % (min_train_loss))

                # Learning Rate Schedular Step
                lr = scheduler_step(LOGGER=LOGGER, CONFIG=CONFIG, schedular=LR_SCHEDULER, lr=lr, val_loss = val_loss)

                # Early Stopping for Overfitting Stopping
                stop, p_counter = early_stop(LOGGER=LOGGER, CONFIG=CONFIG, p_counter=p_counter, training_losses=training_losses)
                LOGGER.log("") # Add New Line
                end_time = time.time()
                logTime(start_time, end_time, logger=LOGGER)

                if epoch == CONFIG.TRAIN_EPOCHS-1 and CONFIG.FINE_TUNE_EPOCHS > 0:
                    LOGGER.log("\t" + "-"*100)
                    LOGGER.log("\t" + "\t\t\tFine Tuning")
                    LOGGER.log("\t" + "-"*100)
                    # Open all layers
                    for _, param in MODEL.named_parameters():
                        param.requires_grad = True
                    fine_tuning = True
                    p_counter = 1
                    if CONFIG.LR_SCHEDULAR != "RLRP":
                        LR_SCHEDULER = ReduceLROnPlateau(optimizer=OPTIMIZER, factor=CONFIG.LRS_FACTOR, patience=CONFIG.LRS_PATIENCE)
                        for param_group in OPTIMIZER.param_groups:
                            param_group['lr'] = 1e-4
                
                elif epoch == total_epochs-1:
                    del MODEL, OPTIMIZER, LR_SCHEDULER
                    LOGGER.log("\t" + "-"*100)
                    LOGGER.log("\t" + f"For Fold [{fold+1}] Testing Model: {CONFIG.PATH_MODEL_SAVE}")
                    # Calculate Performance Metrics
                    MODEL, OPTIMIZER, LR_SCHEDULER = load_model(CONFIG=CONFIG, LOGGER=LOGGER, fold=fold, load_best=True, fineTune=False)
                    MODEL.eval()
                    report = test_model(
                        t_model=MODEL,
                        test_loader = val_loader,
                        test_class_weights = val_class_weights,
                        device = CONFIG.DEVICE,
                        path_save=CONFIG.PATH_PERFORMANCE_SAVE,
                        class_names=CONFIG.CLASS_NAMES,
                        logger = LOGGER
                    )

                    for _class in CONFIG.CLASS_NAMES:
                        metrics = report[_class]
                        precision_values[_class].append(metrics["precision"])
                        recall_values[_class].append(metrics["recall"])
                        f1_values[_class].append(metrics["f1-score"])
                        LOGGER.log(f"\n\t\tClass: {_class}")
                        LOGGER.log(f"\t\t|--- Precision: {metrics['precision']: 0.5f}")
                        LOGGER.log(f"\t\t|--- Recall: {metrics['recall']: 0.5f}")
                        LOGGER.log(f"\t\t|--- F1-Score: {metrics['f1-score']: 0.5f}")

                    del train_loader, val_loader
                    break
                        
            np.savetxt(CONFIG.PATH_LOSSES_SAVE, verify_lengths(training_losses, validation_losses), fmt="%0.5f", delimiter=",")
            
            plot_losses(
                fold=fold,
                training_losses=training_losses, 
                validation_losses=validation_losses,
                save_path=CONFIG.PATH_LOSSPLOT_SAVE, 
                logger=LOGGER
            )
            # fold_min_val_loss.append(min_val_loss)

    except KeyboardInterrupt:
        # Exit Loop code
        LOGGER.log("\t" + "Keyboard Interrupt: Exiting Loop...")
    finally:
        final_values = {}
        for _class in CONFIG.CLASS_NAMES:
            p = torch.tensor(precision_values[_class])
            r = torch.tensor(recall_values[_class])
            f1 = torch.tensor(f1_values[_class])
            final_values[_class] = {
                "precision": precision_values[_class],
                "recall": recall_values[_class],
                "f1-score": f1_values[_class]
            }
            LOGGER.log(f"\tClass: {_class}")
            LOGGER.log(f"\t|--- Precision: mean={p.mean(): 0.5f}, std={p.std(): 0.5f}, median={p.median(): 0.5f}")
            LOGGER.log(f"\t|--- Recall: mean={r.mean(): 0.5f}, std={r.std(): 0.5f}, median={r.median(): 0.5f}")
            LOGGER.log(f"\t|--- F1-Score: mean={f1.mean(): 0.5f}, std={f1.std(): 0.5f}, median={f1.median(): 0.5f}")
        
        with open(CONFIG.PATH_PERFORMANCE_FOLDER + "final_performance.json", "w") as json_file:
            json.dump(final_values, json_file, indent=4)
        result = final_values
        for _class in CONFIG.CLASS_NAMES:
            for score_name in result[_class]:
                scores = result[_class][score_name]
                result[_class][score_name] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "median": np.median(scores)
                }
        with open(CONFIG.PATH_MODEL_FOLDER + "final_result.json" ,"w") as file:
                json.dump(result, file, indent=4)

def train():
    LOGGER.log("\n" + "#"*115 + "\n")
    LOGGER.log("\t\t\t\t\tTraining: " + CONFIG.MODEL_NAME)
    LOGGER.log("\n" + "#"*115)

    # Setting up variables for saving
    training_losses = []
    validation_losses = []
    p_counter = 1
    min_val_loss = float('inf')
    lr = CONFIG.LEARNING_RATE

    epoch = 0
    fine_tuning = False
    total_epochs = CONFIG.TRAIN_EPOCHS + CONFIG.FINE_TUNE_EPOCHS

    # Getting sample weights to use in random sampler and loss calculation
    _, sample_weights_train = get_sample_weights(CONFIG.TRAIN_DATA.dataset, CONFIG.TRAIN_DATA.indices, "Train", logger = LOGGER)
    val_class_weights, sample_weights_val = get_sample_weights(CONFIG.VAL_DATA.dataset, CONFIG.VAL_DATA.indices, "Val", logger = LOGGER)
    
    # Setting up Loss functions
    CONFIG.CRITERION_TRAIN = nn.CrossEntropyLoss()
    CONFIG.CRITERION_VAL = nn.CrossEntropyLoss(weight=val_class_weights.to(CONFIG.DEVICE))
    
    # Training oversampler for class imbalance
    SAMPLER_TRAIN = WeightedRandomSampler(weights=sample_weights_train , num_samples=len(sample_weights_train), replacement=True, generator=CONFIG.GENERATOR)
    
    # Creating Dataloaders
    train_loader = DataLoader(dataset = CONFIG.TRAIN_DATA, batch_size = CONFIG.BATCH_LOAD, num_workers=CONFIG.WORKERS//2, pin_memory=True, sampler=SAMPLER_TRAIN, generator=CONFIG.GENERATOR, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(dataset = CONFIG.VAL_DATA, batch_size = CONFIG.BATCH_LOAD, num_workers=CONFIG.WORKERS//2, pin_memory=True, generator=CONFIG.GENERATOR, persistent_workers=True, prefetch_factor=4)
    
    # Setup model, optimizer and schedular
    MODEL, OPTIMIZER, LR_SCHEDULER = load_model(fold=-1, CONFIG=CONFIG, LOGGER=LOGGER, num_classes=len(list(CONFIG.CLASS_NAMES)))
    try:

        for epoch in range(total_epochs):
            start_time = time.time()
            LOGGER.log("\t" + f"{('--' if not fine_tuning else '++')}" + "-"*98)
            LOGGER.log("\t" + f"EPOCH: [{epoch+1}/{total_epochs}]" + "\t"*8 + f"PERSISTENCE: [{p_counter}/{CONFIG.PERSIST}]")
            LOGGER.log("\t" + f"{('--' if not fine_tuning else '++')}" + "-"*98)

            train_loss = 0.0
            accum_loss = 0.0
            count = 0
            # Training 1 Epoch
            MODEL.train()
            for step, train_XY in enumerate(train_loader, 0):
                
                # Extract X and Y
                imgs = train_XY[0].to(CONFIG.DEVICE)
                labels = train_XY[1].to(CONFIG.DEVICE)
                # imgs = train_XY["image"].to(CONFIG.DEVICE)
                # labels = train_XY["label"].to(CONFIG.DEVICE)
                
                # Predict labels 
                y_pred = MODEL(imgs)

                # Calculate Error
                error = CONFIG.CRITERION_TRAIN(y_pred, labels)
                error.backward()
                accum_loss += error.item()
                
                # Gradient Accumulation
                print("\t" +"\tSTEP: [%d/%d]" % (step+1,len(train_loader)), end= "\r")
                if(count*CONFIG.BATCH_LOAD >= CONFIG.BATCH_SIZE):    
                    OPTIMIZER.step()
                    OPTIMIZER.zero_grad()
                    train_loss += accum_loss
                    print("\t" +"\tSTEP: [%d/%d]\t\t\t\t\t\t>>>>>Batch Loss: [%0.5f]" % (step+1,len(train_loader),accum_loss/(count)), end = "\r") # Print avg batch loss instead of total accum loss
                    accum_loss = 0.0
                    count = 0
                count += 1
            # avg epoch loss
            train_loss /= len(train_loader)
            training_losses.append(train_loss)
            LOGGER.log("\n\n\t" +"\tTraining Loss: [%0.5f]" % (training_losses[-1]))

            # Validation
            MODEL.eval()
            with torch.no_grad():
                val_loss = 0
                for val_XY in val_loader:
                    imgs = val_XY[0].to(CONFIG.DEVICE)
                    labels = val_XY[1].to(CONFIG.DEVICE)
                    # imgs = val_XY["image"].to(CONFIG.DEVICE)
                    # labels = val_XY["label"].to(CONFIG.DEVICE)

                    y_pred = MODEL(imgs)
                    val_loss += CONFIG.CRITERION_VAL(y_pred, labels).item()
            val_loss /= len(val_loader)
            validation_losses.append(val_loss)
            LOGGER.log("\t" +"\tWeighted Val Loss: [%0.5f]" % (val_loss))

            # Save Best Model with minimum validation loss
            p_counter += 1
            if(val_loss < min_val_loss):
                min_val_loss = val_loss
                save_model(CONFIG=CONFIG, model=MODEL, optim=OPTIMIZER, lr_schedular=LR_SCHEDULER, epoch=epoch, train_loss=train_loss)
                p_counter = 1
            LOGGER.log("\t" +"\tMinimum Val Loss: [%0.5f]" % (min_val_loss))

            # Learning Rate Schedular Step
            lr = scheduler_step(LOGGER=LOGGER, schedular=LR_SCHEDULER, lr=lr, val_loss = val_loss)

            # Early Stopping for Overfitting Stopping
            stop, p_counter = early_stop(LOGGER=LOGGER, CONFIG=CONFIG, p_counter=p_counter, training_losses=training_losses)
            LOGGER.log("") # Add New Line
            end_time = time.time()
            logTime(start_time, end_time, logger=LOGGER)

            # After last training epoch, open layers for fine-tuning
            if epoch == CONFIG.TRAIN_EPOCHS-1:
                LOGGER.log("\t" + "-"*100)
                LOGGER.log("\t" + "\t\t\tFine Tuning")
                LOGGER.log("\t" + "-"*100)
                # Open all layers
                for _, param in MODEL.named_parameters():
                    param.requires_grad = True
                fine_tuning = True
                p_counter = 1
                LR_SCHEDULER = ReduceLROnPlateau(optimizer=OPTIMIZER, factor=CONFIG.LRS_FACTOR, patience=CONFIG.LRS_PATIENCE)
                for param_group in OPTIMIZER.param_groups:
                    param_group['lr'] = 1e-4
            
            # After last epoch, clean up stuff
            elif epoch == total_epochs-1:
                del MODEL, OPTIMIZER, LR_SCHEDULER
                del train_loader, val_loader

        # Saving training and validation losses  
        np.savetxt(CONFIG.PATH_LOSSES_SAVE, verify_lengths(training_losses, validation_losses), fmt="%0.5f", delimiter=",")
        
        # plotting losses
        plot_losses(
            fold=None,
            training_losses=training_losses, 
            validation_losses=validation_losses,
            save_path=CONFIG.PATH_LOSSPLOT_SAVE, 
            logger=LOGGER
        )
        # fold_min_val_loss.append(min_val_loss)

    except KeyboardInterrupt:
        # Saving training and validation losses  
        np.savetxt(CONFIG.PATH_LOSSES_SAVE, verify_lengths(training_losses, validation_losses), fmt="%0.5f", delimiter=",")
        
        # plotting losses
        plot_losses(
            fold=None,
            training_losses=training_losses, 
            validation_losses=validation_losses,
            save_path=CONFIG.PATH_LOSSPLOT_SAVE, 
            logger=LOGGER
        )
        # Exit Loop code
        LOGGER.log("\t" + "Keyboard Interrupt: Exiting Loop...")

def test():
    LOGGER.log("\t" + "-"*100)
    LOGGER.log("\t" + f"Testing Model: {CONFIG.PATH_MODEL_SAVE}")
    
    # Performance Metrics
    precision_values = {key: [] for key in CONFIG.CLASS_NAMES}
    recall_values = {key: [] for key in CONFIG.CLASS_NAMES}
    f1_values = {key: [] for key in CONFIG.CLASS_NAMES}

    # Create Test loader
    test_class_weights, test_sample_weights = get_sample_weights(CONFIG.TEST_DATA.dataset, CONFIG.TEST_DATA.indices, "Test", logger = LOGGER)
    test_loader = DataLoader(dataset = CONFIG.TEST_DATA, batch_size = CONFIG.BATCH_LOAD, num_workers=CONFIG.WORKERS//2, pin_memory=True, generator=CONFIG.GENERATOR, persistent_workers=True, prefetch_factor=4)

    # Load best model
    MODEL, OPTIMIZER, LR_SCHEDULER = load_model(fold=-1, CONFIG=CONFIG, LOGGER=LOGGER, load_best=True, fineTune=False, num_classes=len(list(CONFIG.CLASS_NAMES)))
    MODEL.eval()
    report, _, _ = test_model(
        t_model=MODEL,
        test_loader = test_loader,
        test_class_weights = test_class_weights,
        device = CONFIG.DEVICE,
        path_save=CONFIG.PATH_PERFORMANCE_SAVE,
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
    
    with open(CONFIG.PATH_PERFORMANCE_FOLDER + "final_performance.json", "w") as json_file:
        json.dump(report, json_file, indent=4)
  
def test_KCV():
    precision_values = {key: [] for key in CONFIG.CLASS_NAMES}
    recall_values = {key: [] for key in CONFIG.CLASS_NAMES}
    f1_values = {key: [] for key in CONFIG.CLASS_NAMES}
    y_true_folds = []
    y_pred_folds = []
    y_pred_prob_folds = []

    for fold, (train_idx, val_idx) in enumerate(KF.split(CONFIG.TRAIN_DATA)):
        _val = Subset(CONFIG.TRAIN_DATA, val_idx)

        val_class_weights, sample_weights_val = get_sample_weights(CONFIG.TRAIN_DATA, val_idx, "Val", logger = LOGGER)

        
        CONFIG.CRITERION_VAL = nn.CrossEntropyLoss(weight=val_class_weights.to(CONFIG.DEVICE))
        # SAMPLER_TRAIN = WeightedRandomSampler(weights=sample_weights_train , num_samples=len(sample_weights_train), replacement=True, generator=CONFIG.GENERATOR)
        
        # train_loader = DataLoader(dataset = _train, batch_size = CONFIG.BATCH_LOAD, num_workers=CONFIG.WORKERS//2, pin_memory=True, sampler=SAMPLER_TRAIN, generator=CONFIG.GENERATOR, persistent_workers=True, prefetch_factor=4)
        val_loader = DataLoader(dataset = _val, batch_size = CONFIG.BATCH_LOAD, num_workers=CONFIG.WORKERS//2, pin_memory=True, generator=CONFIG.GENERATOR, persistent_workers=True, prefetch_factor=4)
        
        LOGGER.log("\t" + '+'*100)
        LOGGER.log("\t" + f"For Fold [{fold+1}/{CONFIG.K_FOLD}] Testing Model: {CONFIG.PATH_MODEL_SAVE}")
        LOGGER.log("\t" + '+'*100)
        # Calculate Performance Metrics
        MODEL, OPTIMIZER, LR_SCHEDULER = load_model(CONFIG=CONFIG, LOGGER=LOGGER, fold=fold, load_best=True, fineTune=False)
        MODEL.eval()
        report, y_true_labels, y_pred_labels, y_pred_probs = test_model(
            t_model=MODEL,
            test_loader = val_loader,
            test_class_weights = val_class_weights,
            device = CONFIG.DEVICE,
            path_save=CONFIG.PATH_PERFORMANCE_SAVE,
            class_names=CONFIG.CLASS_NAMES,
            logger = LOGGER
        )
        y_true_folds.append(y_true_labels)
        y_pred_folds.append(y_pred_labels)
        y_pred_prob_folds.append(y_pred_probs)

        for _class in CONFIG.CLASS_NAMES:
            metrics = report[_class]
            precision_values[_class].append(metrics["precision"])
            recall_values[_class].append(metrics["recall"])
            f1_values[_class].append(metrics["f1-score"])
            LOGGER.log(f"\n\t\tClass: {_class}")
            LOGGER.log(f"\t\t|--- Precision: {metrics['precision']: 0.5f}")
            LOGGER.log(f"\t\t|--- Recall: {metrics['recall']: 0.5f}")
            LOGGER.log(f"\t\t|--- F1-Score: {metrics['f1-score']: 0.5f}")

        del val_loader
        del MODEL, OPTIMIZER, LR_SCHEDULER
    
    final_values = {}
    for _class in CONFIG.CLASS_NAMES:
        p = torch.tensor(precision_values[_class])
        r = torch.tensor(recall_values[_class])
        f1 = torch.tensor(f1_values[_class])
        final_values[_class] = {
            "precision": precision_values[_class],
            "recall": recall_values[_class],
            "f1-score": f1_values[_class]
        }
        LOGGER.log(f"\tClass: {_class}")
        LOGGER.log(f"\t|--- Precision: mean={p.mean(): 0.5f}, std={p.std(): 0.5f}, median={p.median(): 0.5f}")
        LOGGER.log(f"\t|--- Recall: mean={r.mean(): 0.5f}, std={r.std(): 0.5f}, median={r.median(): 0.5f}")
        LOGGER.log(f"\t|--- F1-Score: mean={f1.mean(): 0.5f}, std={f1.std(): 0.5f}, median={f1.median(): 0.5f}")
    
    with open(CONFIG.PATH_PERFORMANCE_FOLDER + "final_performance.json", "w") as json_file:
        json.dump(final_values, json_file, indent=4)
    
    make_confusion_matrix(
        y_true_folds=y_true_folds,
        y_pred_folds=y_pred_folds,
        n_classes=len(CONFIG.CLASS_NAMES),
        class_names=CONFIG.CLASS_NAMES,
        path_save=CONFIG.PATH_PERFORMANCE_FOLDER + "confusion_matrix.png"
    )
    make_roc_curve_macro(
        y_true_folds=y_true_folds,
        y_prob_folds=y_pred_prob_folds,
        n_classes=len(CONFIG.CLASS_NAMES),
        class_names=CONFIG.CLASS_NAMES,
        path_save=CONFIG.PATH_PERFORMANCE_FOLDER + "roc_curve.png"
    )
    
    result = final_values
    for _class in CONFIG.CLASS_NAMES:
        for score_name in result[_class]:
            scores = result[_class][score_name]
            result[_class][score_name] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "median": np.median(scores)
            } 
    with open(CONFIG.PATH_MODEL_FOLDER + "final_result.json" ,"w") as file:
        json.dump(result, file, indent=4)
    
    with open(CONFIG.PATH_MODEL_FOLDER + "final_report.json" ,"w") as file:
        json.dump(report, file, indent=4)



if __name__ == "__main__":
    CONFIG = Config()
    LOGGER = MyLogger(
        server_url = CONFIG.SERVER_URL,
        server_username = CONFIG.SERVER_USERNAME,
        server_folder = CONFIG.SERVER_FOLDER,
        model_name = CONFIG.MODEL_NAME,
        path_localFile = CONFIG.PATH_MODEL_LOG_FILE
    )

    OPTIMIZER: torch.optim.SGD = None
    LR_SCHEDULER: CosineAnnealingWarmRestarts = None
    SAMPLER: WeightedRandomSampler = None
    MODEL: torch.nn.Module = None

    LOGGER.log("\n\n" + "="*54 + " START " + "="*54)
    LOGGER.log(f"Strategy: {'K-Fold' if CONFIG.K_FOLD > 0 else 'Simple'}")
    LOGGER.log(f"Training Epochs: {CONFIG.TRAIN_EPOCHS}")
    LOGGER.log(f"Fine Tuning Epochs: {CONFIG.FINE_TUNE_EPOCHS}")
    LOGGER.log(f"Using GPU: {CONFIG.DEVICE}")
    LOGGER.log(f"Batch Size: {CONFIG.BATCH_SIZE}")
    LOGGER.log(f"Learning Rate: {CONFIG.LEARNING_RATE}")
    LOGGER.log(f"Early Stopping with Persistence: {CONFIG.PERSIST}")
    # Add if statement
    if CONFIG.LR_SCHEDULAR == "CAWR":
        LOGGER.log(f"LR Schedular: CosineAnnealingWarmRestarts + ReduceLROnPlataue") 
    elif CONFIG.LR_SCHEDULAR == "RLRP":
        LOGGER.log(f"LR Schedular: ReduceLROnPlataue")
        LOGGER.log(f"|---Patience: {CONFIG.LRS_PATIENCE}")
        LOGGER.log(f"|---Factor: {CONFIG.LRS_FACTOR}")
    else:
        LOGGER.log(f"Invalid LR Schedular => {CONFIG.LR_SCHEDULAR}")
    if CONFIG.K_FOLD > 0:
        KF = KFold(n_splits=CONFIG.K_FOLD, shuffle=True, random_state=CONFIG.RANDOM_STATE)
        # train_KCV()
        test_KCV()
    else:
        # Train test split
        CONFIG.TRAIN_DATA, CONFIG.VAL_DATA, CONFIG.TEST_DATA = random_split(dataset=CONFIG.DATA, lengths=[0.7, 0.1, 0.2], generator=CONFIG.GENERATOR)
        CONFIG.VAL_DATA.transform = CONFIG.TRANSFORMS_TEST
        CONFIG.TEST_DATA.transform = CONFIG.TRANSFORMS_TEST
        train()
        test()
    
    LOGGER.log("\n\n" + "="*55 + " END " + "="*55 + "\n\n")