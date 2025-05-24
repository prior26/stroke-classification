from torch import nn

class MainNet(nn.Module):
    def __init__(self, model_name: str, model: nn.Module, feature_dim: int, last_freezed_layer: str, input_size: tuple):
        super(MainNet, self).__init__()
        self.model_name: str = model_name
        self.model: nn.Module = model
        self.feature_dim: int = feature_dim
        self.last_freezed_layer: str = last_freezed_layer
        self.input_size: tuple = input_size

    def forward(self, x):
        return self.model(x)
    
    def get_last_freezed_layer(self):
        return self.last_freezed_layer
    
    def get_feature_dim(self):
        return self.feature_dim
    
    def get_target_module(self, target_module_name):
        for name, module in self.model.named_children():
            if target_module_name in name:
                return module
        raise ValueError(f"Invalid target module for {self.model_name}")
            