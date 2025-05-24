import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torch import nn
from torchinfo import summary
from torchvision import models
from torch.nn import functional as f
from Models.ConvNets import ResNet
from Models.TransNets import SWIN, CvT
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

from Logger import MyLogger

class StrokeDetector(nn.Module):
    
    def __init__(
            self,
            model_size = "s",
            input_size = (1, 1, 224, 224),
            num_classes = 3,
            embed_dim = 256,
            fold = -1,
            strategy = "FCBW",
            logger:MyLogger=None,
            device = "cpu",
            verbose:bool = True

    ):
        super(StrokeDetector, self).__init__()
        self.device = device
        self.backbone_names = ["ResNet", "CvT", "SWIN"]
        self.stacked_id = {
            0: "ResNet",
            1: "CvT",
            2: "SWIN"
        }
        self.verbose = verbose
        self.first_val = True
        self.strategy = strategy
        self.logger = logger if logger is not None else MyLogger()
        self.logger.log("")
        
        # Layers to break models at
        resnet_layer = "layer4" 
        cvt_layer = "blocks.11"
        swin_layer = "features.7"

        # Models Should be pretrained
        num_heads = 16

        # Initialize the feature extractors
        self.resnet_extractor = self._init_resnet(
            model_size=model_size, 
            input_size=input_size,
            num_classes=num_classes,
            target_layer=resnet_layer, 
            fold=fold, 
            device=device
        )
        self.swin_extractor = self._init_swin(
            model_size=model_size, 
            input_size=input_size,
            num_classes=num_classes,
            target_layer=swin_layer, 
            fold=fold, 
            device=device
        )
        self.cvt_extractor = self._init_cvt(
            model_size=model_size, 
            input_size=input_size,
            num_classes=num_classes,
            target_layer=cvt_layer, 
            fold=fold, 
            device=device
        )

        # Get feature dimensions from each extractor
        self.resnet_feat_dim = self._get_feature_dim(self.resnet_extractor, input_size)
        self.cvt_feat_dim = self._get_feature_dim(self.cvt_extractor, input_size)
        self.swin_feat_dim = self._get_feature_dim(self.swin_extractor, input_size)
        
        self.resnet_projection = nn.Linear(self.resnet_feat_dim, embed_dim)
        self.cvt_projection = nn.Linear(self.cvt_feat_dim, embed_dim)
        self.swin_projection = nn.Linear(self.swin_feat_dim, embed_dim)

        if strategy == "FCBW":
            # Adding Layers for final classification
            self.backbone_weights = nn.Parameter(torch.ones(3, 1) / 3.0, requires_grad=True)  # Initialize to equal weights
            final_dim = embed_dim*3
        
        elif strategy == "QCA" or strategy == "QSA":
            self.model_embeddings = nn.Parameter(torch.randn(3, embed_dim))
            self.gate = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim//2),
                nn.GELU(),
                nn.Linear(embed_dim//2, embed_dim//4),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(embed_dim//4, 1)
            )
            self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=0.4)
            final_dim = embed_dim
        else:
            raise ValueError(f"Fusion method must be one of ['FCBW', 'QCA', 'QSA'], but got {strategy}")
        
        # Construct the classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(final_dim, 128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )
        self.to(device)
        self.logger.log(f"\n\tFeature Dims")
        self.logger.log(f"\t|---> ResNet: {self.resnet_feat_dim}")
        self.logger.log(f"\t|---> CvT: {self.cvt_feat_dim}")
        self.logger.log(f"\t|---> SWIN: {self.swin_feat_dim}\n")

    def _init_resnet(self, model_size, input_size,num_classes, target_layer, fold, device):
        """Initialize ResNet feature extractor"""
        resnet = ResNet(
            model_size=model_size,
            input_size=input_size,
            num_classes=num_classes,
            extractLayer = target_layer,
            fold=fold,
            use_as_feature_extractor=True,
            device=device,
            logger=self.logger
        )
        return resnet.model.to(self.device)
    
    def _init_cvt(self, model_size, input_size,num_classes, target_layer, fold, device):
        """Initialize CvT feature extractor"""
        cvt = CvT(
            model_size=model_size,
            input_size=input_size,
            num_classes=num_classes,
            extractLayer = target_layer,
            fold=fold,
            use_as_feature_extractor=True,
            device=device,
            logger=self.logger
        )
        return cvt.model.to(self.device)
    
    def _init_swin(self, model_size, input_size,num_classes, target_layer, fold, device):
        """Initialize SWIN feature extractor"""
        swin = SWIN(
            model_size=model_size,
            input_size=input_size,
            num_classes=num_classes,
            extractLayer = target_layer,
            fold=fold,
            use_as_feature_extractor=True,
            device=device,
            logger=self.logger
        )
        return swin.model.to(self.device)
    
    def _get_feature_dim(self, model, input_size):
        """Get feature dimension by passing a dummy input"""
        dummy_input = torch.randn(input_size).to(self.device)
        model = self.move_to_device(model, self.device)
        with torch.no_grad():
            output = model(dummy_input)
            return output.size(1)  # Get the feature dimension
    
    def move_to_device(self, model: nn.Module, device: torch.device):
        for param in model.parameters():
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        for buffer in model.buffers():
            buffer.data = buffer.data.to(device)
        model.to(device)
        return model
    
    def forward(self, x):
        # Get features from each extractor
        resnet_features = self.resnet_extractor(x)
        cvt_features = self.cvt_extractor(x)
        swin_features = self.swin_extractor(x)
        
        if self.strategy == "FCBW":
            # Weighing the Backbones
            weights = self.backbone_weights
            backbone_weights = F.softmax(weights, dim=0)  # [3, 1]
            resnet_proj = self.resnet_projection(resnet_features) * backbone_weights[0]
            cvt_proj = self.cvt_projection(cvt_features) * backbone_weights[1]  
            swin_proj = self.swin_projection(swin_features) * backbone_weights[2]

            # Print the weights during evaluation for insight
            self._eval_print(backbone_weights=backbone_weights)
            combined_features = torch.cat([resnet_proj, cvt_proj, swin_proj], dim=1)
        
        elif self.strategy == "QCA" or self.strategy == "QSA":
            # Get Projection from each model
            resnet_proj = self.resnet_projection(resnet_features).unsqueeze(1)
            cvt_proj = self.cvt_projection(cvt_features).unsqueeze(1)
            swin_proj = self.swin_projection(swin_features).unsqueeze(1)

            # Stack projections
            stacked = torch.cat([resnet_proj, cvt_proj, swin_proj], dim=1)

            # Add model embeddings for model understanding
            modality_embed = self.model_embeddings.unsqueeze(0)  # [1, 3, D]
            features = stacked + modality_embed

            # Get Query selection index
            gate_logits = self.gate(features)
            gate_weights = F.softmax(gate_logits, dim=1)

            self._eval_print(batch_size=x.shape[0], gate_weights=gate_weights)

            if self.strategy == "QCA":
                query = torch.sum(features * gate_weights, dim=1)  # [B, D]
                query = query.unsqueeze(1)  # [B, 1, D]
                key_value = features
            elif self.strategy == "QSA":
                query_indices = torch.argmax(gate_weights, dim=1)
                query = []
                key_value = []
                for i in range(x.size(0)):
                    q_idx = query_indices[i].item()
                    query.append(features[i, q_idx:q_idx+1])  # [1, D]
                    kv = torch.cat([features[i, j:j+1] for j in range(3) if j != q_idx], dim=0)  # [2, D]
                    key_value.append(kv.unsqueeze(0))  # [1, 2, D]
                query = torch.cat(query, dim=0).unsqueeze(1)
                key_value = torch.cat(key_value, dim=0)

            # Apply MHA
            attn_output, _ = self.attention(query=query, key=key_value, value=key_value)
            combined_features = attn_output.squeeze(1)
        
        # Pass through classifier
        output = self.classifier(combined_features)
        return output
    
    def _eval_print(self, **kwargs):
        if self.verbose:
            if not self.training and self.first_val:
                self.first_val = False
                if self.strategy == "FCBW":
                    weight_dict = {name: round(float(weight), 4) for name, weight in zip(self.backbone_names, kwargs["backbone_weights"])}
                    self.logger.log(f"\t\tGlobal Backbone Weights: {weight_dict}")
                elif self.strategy == "QSA":
                    query_indices = torch.argmax(kwargs["gate_weights"], dim=1)  # [B]
                    counts = []
                    for i in range(kwargs["batch_size"]):
                        q_idx = query_indices[i].item()
                        counts.append(self.stacked_id[q_idx])
                    counts = Counter(counts)
                    self.logger.log(f"\t\tSelection Counts: [{dict(counts)}]")
                elif  self.strategy == "QCA":
                    avg_weights = kwargs["gate_weights"].mean(dim=0)  # Shape: [N]

                    # Map average weights to stacked model names
                    weights_dict = {self.stacked_id[i]: avg_weights[i].item() for i in range(avg_weights.shape[0])}
                    self.logger.log(f"\t\tAverage Selection Weights: {weights_dict}")
                    
            else:
                self.first_val = True
                # if self.strategy == "FCBW":
                    # weight_dict = {name: round(float(weight), 4) for name, weight in zip(self.backbone_names, kwargs["backbone_weights"])}
                    # print(f"\t\t>>> Global Backbone Weights: {weight_dict}")
        else:
            return
        
# Example usage
if __name__ == "__main__":
    # Create the model
    model = StrokeDetector(
        model_size = "s",
        input_size = (1, 1, 224, 224),
        num_classes = 3,
        embed_dim = 256,
        fold = -1,
        strategy = "FCBW",
        device = "cpu",
        verbose= True
    )
    summary(model, input_size=(1, 1, 224, 224), depth=3, col_names=["input_size","output_size","num_params"], verbose=1, device="cpu")

    # Test with dummy input
    dummy_input = torch.randn(1, 1, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")