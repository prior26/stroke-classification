import torch
from torch import nn
from torchinfo import summary
from torchvision import models
from torch.nn import functional as f
import timm
from Models.MainNet import MainNet
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Logger import MyLogger

class CvT(MainNet):
    def __init__(
            self, 
            model_size:str = "b", 
            input_size=(1, 1, 224, 224), 
            num_classes=3, 
            freezeToLayer: str = "blocks.5", 
            extractLayer="blocks.5", 
            pretrained:bool = True, 
            use_as_feature_extractor:bool=False,
            fold:int = -1, 
            device:str = "cpu",
            logger:MyLogger=None
    ): 
        
        model:nn.Module
        self.device = device
        try:
            if model_size == "t":
                model = timm.create_model(model_name="convit_tiny.fb_in1k", pretrained=pretrained, in_chans=input_size[1], num_classes = num_classes)
            elif model_size == "s":
                model = timm.create_model(model_name="convit_small.fb_in1k", pretrained=pretrained, in_chans=input_size[1], num_classes = num_classes)
            elif model_size == "b":
                model = timm.create_model(model_name="convit_base.fb_in1k", pretrained=pretrained, in_chans=input_size[1], num_classes = num_classes)
            else:
                raise ValueError(f"Model Size should be from [t, s, b] i.e. tiny, small, base, but {model_size} was passed")
        except ValueError as e:
            logger.log(f"Error: {e}")
            sys.exit(1)

        if use_as_feature_extractor:
            if fold < 1:
                if num_classes == 3:
                    path = f"Classifiers/Transformer Networks/CvT_{model_size.capitalize()}_new/Simple/Checkpoint.pth"
                elif num_classes == 2:
                    path = f"Classifiers/Transformer Networks/CvT_{model_size.capitalize()}/Simple/Checkpoint.pth"
                logger.log(f"\tLoading CvT Simple Model from: {path}")    
            else:
                path = f"Classifiers/Transformer Networks/CvT_{model_size.capitalize()}/K-Fold/F{str(fold)}_Checkpoint.pth"
                logger.log(f"\tLoading CvT Model from Fold: {str(fold)}")

            checkpoint = torch.load(f=path, map_location=self.device, weights_only=True)
            state_dict = {k.replace("model.", "") : v for k, v in checkpoint["model_state_dict"].items()}
            model.load_state_dict(state_dict)
            model = self.get_feature_extractor(model, target_module_name=extractLayer)

            # Estimate feature_dim by passing a dummy input
            dummy_input = torch.randn(input_size, device=self.device)
            with torch.no_grad():
                output:torch.Tensor = model(dummy_input)
                feature_dim = output.size()
            
            # Freeze Feature Extractor
            for name, param in model.named_parameters():
                param.requires_grad = False
                last_freezed_layer = name
        
        else:
            logger.log("\nFreezing CvT Layers...")
            # Freeze Specified Layers
            last_freezed_layer = ""
            if (freezeToLayer is not None and freezeToLayer != "") or freezeToLayer == "all":
                for name, param in model.named_parameters():
                    param.requires_grad = False
                    if "patch_embed" in name:
                        logger.log("|---> Found First Layers, Leaving Trainable...")
                        param.requires_grad = True
                    elif freezeToLayer in name:
                        last_freezed_layer = name
                        break
            feature_dim = model.head.in_features

        super(CvT, self).__init__(
            model_name="CvT",
            model=model,
            feature_dim=feature_dim,
            last_freezed_layer=last_freezed_layer,
            input_size = input_size
        )

    
    def get_feature_extractor(self, full_model:nn.Module, target_module_name="blocks.11"):
        new_model = nn.Sequential()
        name_target_module, name_target_submodule = target_module_name.split(".")
        found = False
        
        for name, module in full_model.named_children():
            if name == name_target_module and isinstance(module, nn.ModuleList):
                # Loop through submodules inside ModuleList
                for i, submodule in enumerate(module):
                    new_model.add_module(f"{name}_{i}", submodule)
                    if str(i) == name_target_submodule:
                        found = True
                        break
                if found:
                    break
            else:
                new_model.add_module(name, module)
                if name == name_target_module:
                    found = True
                    break

        if not found:
            raise ValueError(f"Target module '{target_module_name}' not found in the model.")
        new_model.add_module("permute", Permute(0, 2, 1))
        new_model.add_module("adaptive_pool", nn.AdaptiveAvgPool2d((432, 1)))
        new_model.add_module("flatten", nn.Flatten())
        return new_model.to(self.device)

class SWIN(MainNet):
    def __init__(
            self, 
            model_size:str='s', 
            input_size=(1, 1, 224, 224), 
            num_classes=3, 
            freezeToLayer: str = "features.5.0", 
            extractLayer="features.5", 
            pretrained:bool=True, 
            use_as_feature_extractor:bool=False, 
            fold:int = -1,
            device:str = "cpu",
            logger:MyLogger=None
        ): 
        model: nn.Module
        self.device = device

        try:
            if model_size == 'b':
                model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1 if pretrained else None)
            elif model_size == 's':
                model = models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1 if pretrained else None)
            elif model_size == 't':
                model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
            else:
                raise ValueError(f"Model Size should be from [t, s, b] i.e. tiny, small, base, but {model_size} was passed")
        except ValueError as e:
            logger.log(f"Error: {e}")
            sys.exit(1)
        
        # Modify the input layer for single-channel images if needed
        model.features[0][0] = nn.Conv2d(input_size[1], model.features[0][0].out_channels, 
                                              kernel_size=4, stride=4, padding=0)

        # Modify the final fully connected layer to match the number of output classes
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)


        if use_as_feature_extractor:
            if fold < 1:
                if num_classes == 3:
                    path = f"Classifiers/Transformer Networks/SWIN_{model_size.capitalize()}_new/Simple/Checkpoint.pth"
                elif num_classes == 2:
                    path = f"Classifiers/Transformer Networks/SWIN_{model_size.capitalize()}/Simple/Checkpoint.pth"
                logger.log(f"\tLoading SWIN Simple Model from: {path}")
                
            else:
                path = f"Classifiers/Transformer Networks/SWIN_{model_size.capitalize()}/K-Fold/F{str(fold)}_Checkpoint.pth"
                logger.log(f"\tLoading SWIN Model from Fold: {str(fold)}")

            checkpoint = torch.load(f=path, map_location=self.device, weights_only=True)
            state_dict = {k.replace("model.", "") : v for k, v in checkpoint["model_state_dict"].items()}
            model.load_state_dict(state_dict)
            model = self.get_feature_extractor(model, target_module_name=extractLayer)
            
            # Estimate feature_dim by passing a dummy input
            dummy_input = torch.randn(input_size).to(self.device)
            with torch.no_grad():
                output:torch.Tensor = model(dummy_input)
                feature_dim = output.size()
            
            # Freeze Feature Extractor
            for name, param in model.named_parameters():
                param.requires_grad = False
                last_freezed_layer = name
                
        else:
            feature_dim = model.head.in_features
            # Optionally freeze layers up to a certain layer
            logger.log("\nFreezing SWIN Layers...")
            last_freezed_layer = ""
            if freezeToLayer is not None and freezeToLayer != "":
                for name, param in model.named_parameters():
                    param.requires_grad = False
                    if "features.0" in name:  # Similar to patch_embed in CvT
                        logger.log("\t|---> Found First Layers, Leaving Trainable...")
                        param.requires_grad = True
                    elif freezeToLayer in name:
                        last_freezed_layer = name
                        break
        
        super(SWIN, self).__init__(
            model_name="SWIN",
            model=model,
            feature_dim=feature_dim,
            last_freezed_layer=last_freezed_layer,
            input_size=input_size
        )
        self.model.to(self.device)
    
    def get_feature_extractor(self, full_model:nn.Module, target_module_name="features.7"):
        new_model = nn.Sequential()
        # Parse target module name (format: "features.7")
        name_target_module, name_target_submodule = target_module_name.split(".")
        found = False
        
        for name, module in full_model.named_children():
            if name == name_target_module and isinstance(module, nn.Sequential):
                # For SWIN, the features is a Sequential module with submodules
                for i, submodule in enumerate(module):
                    new_model.add_module(f"{name}_{i}", submodule)
                    if str(i) == name_target_submodule:
                        found = True
                        break
                if found:
                    break
            else:
                new_model.add_module(name, module)
                if name == name_target_module:
                    found = True
                    break
                    
        if not found:
            raise ValueError(f"Target module '{target_module_name}' not found in the model.")
        
        new_model.add_module("permute", Permute(0, 3, 1, 2))
        new_model.add_module("adaptive_pool", nn.AdaptiveAvgPool2d((1, 1)))
        new_model.add_module("flatten", nn.Flatten())
        return new_model.to(self.device)

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

if __name__ == "__main__":
    from torchinfo import summary
    model = CvT(model_size="s", pretrained=True, use_as_feature_extractor=True)
    model.to("cuda:0")
    # for name, param in model.model.named_parameters():
        # print(f"NAME: {name}{'(Freezed Till Here)' if model.get_last_freezed_layer() in name else ''}")
    summary(model, input_size=(1, 1, 224, 224), depth=3, col_names=["input_size","output_size","num_params"], verbose=1, device="cuda:0")
    for name, param in model.named_parameters():
        if param.device == "cpu":
            print(f"{name}: {param.device}")

