import torch
from torch import nn
from torchinfo import summary
from torchvision import models
from Models.MainNet import MainNet
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Logger import MyLogger

class ResNet(MainNet):
    def __init__(
            self, 
            model_size:str = "b", 
            input_size=(1, 1, 224, 224), 
            num_classes=3, 
            freezeToLayer:str = "layer3.12", 
            pretrained=True,
            extractLayer:str = "layer3.12",
            use_as_feature_extractor:bool=False, 
            fold:int = -1,
            device:str = "cpu",
            logger:MyLogger=None
    ):
        """
        Initialize a ResNet model.
        
        Args:
            model_size: Size of the model ('t' for tiny (ResNet50), 's' for small (ResNet101), 'b' for base (ResNet152))
            input_size: Input tensor dimensions (batch, channels, height, width)
            num_classes: Number of output classes
            freezeToLayer: Freeze layers up to this layer (if not None or empty)
            pretrained: Whether to use pretrained weights
            use_as_feature_extractor: Whether to use the model as a feature extractor
        """
        model: nn.Module
        self.device = device

        try:
            # Load the appropriate ResNet model based on model_size
            if model_size == "t":
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            elif model_size == "s":
                model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
            elif model_size == "b":
                model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1 if pretrained else None)
            else:
                raise ValueError(f"Model Size should be from [t, s, b] i.e. tiny (ResNet50), small (ResNet101), base (ResNet152), but {model_size} was passed")
        except ValueError as e:
            logger.log(f"Error: {e}")
            sys.exit(1)
        
        # Modify the input layer for single-channel images if needed
        model.conv1 = nn.Conv2d(input_size[1], 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.bn1.requires_grad_(True)
        
        # Modify the final fully connected layer to match the number of output classes
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        if use_as_feature_extractor:
            if fold < 1:
                if num_classes == 3:
                    path = f"Classifiers/Convolutional Networks/ResNet_{model_size.capitalize()}_new/Simple/Checkpoint.pth"
                elif num_classes == 2: 
                    path = f"Classifiers/Convolutional Networks/ResNet_{model_size.capitalize()}/Simple/Checkpoint.pth"
                logger.log(f"\tLoading ResNet Simple Model from: {path}")
                
            else:
                path = f"Classifiers/Convolutional Networks/ResNet_{model_size.capitalize()}/K-Fold/F{str(fold)}_Checkpoint.pth"
                logger.log(f"\tLoading ResNet Model from Fold: {str(fold)}")

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
            feature_dim = model.fc.in_features
            # Optionally freeze layers up to a certain layer
            logger.log("\nFreezing ResNet Layers...")
            last_freezed_layer = ""
            if freezeToLayer is not None and freezeToLayer != "":
                for name, param in model.named_parameters():
                    param.requires_grad = False
                    if ("conv1.weight" == name or "bn1.weight" == name or "bn1.bias" == name):  # Always keep first conv and bn trainable
                        logger.log("\t|---> Found First Layers, Leaving Trainable...")
                        param.requires_grad = True
                    elif freezeToLayer in name:
                        last_freezed_layer = name
                        break

        super(ResNet, self).__init__(
            model_name="ResNet",
            model=model,
            feature_dim=feature_dim,
            last_freezed_layer=last_freezed_layer,
            input_size=input_size
        )
        model.to(self.device)
    
    def get_feature_extractor(self, full_model:nn.Module, target_module_name="layer4"):
        new_model = nn.Sequential()
        found = False
        
        # For ResNet, we need to handle differently if target includes a submodule number
        # For example, "layer3.12" refers to the 12th block of layer3
        if "." in target_module_name:
            name_target_module, name_target_submodule = target_module_name.split(".")
            name_target_submodule = int(name_target_submodule)
            
            # Start building the new model with the initial layers
            new_model.add_module("layer0.0.conv1", full_model.conv1)
            new_model.add_module("layer0.0.bn1", full_model.bn1)
            new_model.add_module("layer0.0.relu", full_model.relu)
            new_model.add_module("layer0.0.maxpool", full_model.maxpool)
            
            # Add layers until we reach the target layer
            for name, module in full_model.named_children():
                if name == name_target_module:
                    # For layer modules that are ModuleList or Sequential, we need to handle submodules
                    if isinstance(module, nn.Sequential):
                        layer_sequential = nn.Sequential()
                        for i, block in enumerate(module):
                            layer_sequential.add_module(f"{i}", block)
                            if i == name_target_submodule:
                                found = True
                                break
                        new_model.add_module(name, layer_sequential)
                        if found:
                            break
                elif name in ["conv1", "bn1", "relu", "maxpool"]:
                    # Already added
                    continue
                else:
                    new_model.add_module(name, module)
        else:
            # If target is just a layer name without submodule index
            for name, module in full_model.named_children():
                new_model.add_module(name, module)
                if name == target_module_name:
                    found = True
                    break
                    
        if not found:
            raise ValueError(f"Target module '{target_module_name}' not found in the model.")
            
        # Remove the final FC layer if it exists in the new model
        if hasattr(new_model, 'fc'):
            new_model = nn.Sequential(*list(new_model.children())[:-1])
            
        # Add adaptive pooling to ensure fixed output size
        new_model.add_module("adaptive_pool", nn.AdaptiveAvgPool2d((1, 1)))
        new_model.add_module("flatten", nn.Flatten())
            
        return new_model.to(self.device)

# Example usage
if __name__ == "__main__":
    model = ResNet(model_size="s", pretrained=False, use_as_feature_extractor=True)
    summary(model, input_size=(1, 1, 224, 224), depth=4, col_names=["input_size","output_size","num_params"], device="cpu")
    # for name, param in model.model.named_parameters():
    #     print(f"NAME: {name}")
    # Test with a dummy input
    # dummy_input = torch.randn(1, 1, 224, 224)  # Batch size 1, 1 channel, 224x224 image
    # output = model(dummy_input)
    # print(output.shape)  # Should output torch.Size([1, 3])
