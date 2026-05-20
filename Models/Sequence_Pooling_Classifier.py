import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights


class AlexNetEncoder(nn.Module):
    """AlexNet-based visual feature extractor."""
    
    def __init__(self, input_channels=3, use_pretrain=True):
        super().__init__()
        
        # Load AlexNet with or without pretrained weights
        if use_pretrain:
            self.alexnet = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
            print("AlexNet: ImageNet pretrained weights")
        else:
            self.alexnet = alexnet(weights=None)
            print("AlexNet: random initialisation")
        
        # Use only the convolutional feature extractor
        self.features = self.alexnet.features
        
        # Adapt the first conv layer if input channels differ from 3
        if input_channels != 3:
            old_conv = self.features[0]
            self.features[0] = nn.Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding
            )
        
        # AlexNet conv5 outputs 256 feature maps
        self.feature_dim = 256
        
        # Global average pooling to collapse spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        """
        Args:
            x: [batch, channels, H, W]
        Returns:
            features: [batch, 256] global-pooled feature vector
        """
        # Pass through AlexNet convolutional layers
        features = self.features(x)
        
        # Global average pool and flatten
        features = self.global_pool(features)
        features = features.flatten(1)
        
        return features


class SequencePoolingClassifier(nn.Module):
    """Sequence pooling classifier — encodes each frame independently then pools over time."""
    
    def __init__(self, 
                 use_pretrain=True,
                 input_channels=3,
                 pooling_strategy='mean',
                 hidden_dim=128,
                 dropout=0.1,
                 num_classes=11):
        """
        Initialise the sequence pooling classifier.
        
        Args:
            use_pretrain: whether to load ImageNet pretrained weights for AlexNet
            input_channels: number of input image channels
            pooling_strategy: how to aggregate frame features ('mean', 'max', 'last')
            hidden_dim: hidden dimension of the classification head
            dropout: dropout ratio
            num_classes: number of output classes
        """
        super().__init__()
        
        self.use_pretrain = use_pretrain
        self.pooling_strategy = pooling_strategy
        self.num_classes = num_classes
        
        # Validate pooling strategy
        assert pooling_strategy in ['mean', 'max', 'last'], \
            f"pooling_strategy must be 'mean', 'max', or 'last', got {pooling_strategy}"
        
        # Visual encoder — AlexNet backbone, shared across all frames
        self.visual_encoder = AlexNetEncoder(
            input_channels=input_channels,
            use_pretrain=use_pretrain
        )
        
        # Classification head
        feature_dim = self.visual_encoder.feature_dim  # 256
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        print(f"SequencePoolingClassifier configuration:")
        print(f"  AlexNet pretrained: {use_pretrain}")
        print(f"  Pooling strategy: {pooling_strategy}")
        print(f"  Visual feature dim: {feature_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num classes: {num_classes}")
    
    def forward(self, x):
        """
        Forward pass over an image sequence.
        
        Args:
            x: [batch, seq_len, channels, H, W]
        
        Returns:
            logits: [batch, num_classes]
        """
        batch_size, seq_len = x.shape[:2]
        
        # Reshape: [B, S, C, H, W] -> [B*S, C, H, W]
        x_flat = x.view(batch_size * seq_len, *x.shape[2:])
        
        # Encode all frames in parallel: [B*S, 256]
        features_flat = self.visual_encoder(x_flat)
        
        # Reshape back to sequence: [B*S, 256] -> [B, S, 256]
        features = features_flat.view(batch_size, seq_len, -1)
        
        # Pool across the temporal dimension
        if self.pooling_strategy == 'mean':
            # Average over all frames
            pooled_features = torch.mean(features, dim=1)  # [B, 256]
        elif self.pooling_strategy == 'max':
            # Element-wise max over all frames
            pooled_features = torch.max(features, dim=1)[0]  # [B, 256]
        elif self.pooling_strategy == 'last':
            # Use the final frame only
            pooled_features = features[:, -1, :]  # [B, 256]
        else:
            raise ValueError(f"Unknown pooling_strategy: {self.pooling_strategy}")
        
        # Classify the pooled representation
        logits = self.classifier(pooled_features)
        
        return logits
    
    def get_model_info(self):
        """Return a dict summarising the model configuration."""
        return {
            'model_type': 'SequencePoolingClassifier',
            'visual_encoder': 'AlexNet',
            'pretrained': self.use_pretrain,
            'pooling_strategy': self.pooling_strategy,
            'num_classes': self.num_classes,
            'has_temporal_modeling': False,
            'tasks': ['classification']
        }


def create_sequence_pooling_model(num_classes=11, 
                                  use_pretrain=True, 
                                  input_channels=3,
                                  pooling_strategy='mean'):
    """
    Factory function to instantiate a SequencePoolingClassifier.
    
    Args:
        num_classes: number of output classes
        use_pretrain: whether to load ImageNet pretrained weights for AlexNet
        input_channels: number of input image channels
        pooling_strategy: temporal pooling method ('mean', 'max', 'last')
    """
    model = SequencePoolingClassifier(
        use_pretrain=use_pretrain,
        input_channels=input_channels,
        pooling_strategy=pooling_strategy,
        hidden_dim=128,
        dropout=0.1,
        num_classes=num_classes
    )
    
    pretrain_str = "pretrained" if use_pretrain else "random init"
    print(f"Model created — AlexNet ({pretrain_str}) - {pooling_strategy} pooling")
    
    return model


# Usage example
if __name__ == "__main__":
    print("=== SequencePoolingClassifier Test ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  # force CPU for testing
    print(f"Device: {device}\n")
    
    # Test combinations: pretrain flag × pooling strategy
    test_configs = [
        (False, 'mean', "no pretrain + mean pooling"),
        (True, 'mean',  "pretrained + mean pooling"),
        (True, 'max',   "pretrained + max pooling"),
        (True, 'last',  "pretrained + last-frame pooling"),
    ]
    
    for use_pretrain, pool_strategy, desc in test_configs:
        print(f"\n{'='*60}")
        print(f"Configuration: {desc}")
        print('='*60)
        
        model = create_sequence_pooling_model(
            num_classes=11,
            use_pretrain=use_pretrain,
            input_channels=3,
            pooling_strategy=pool_strategy
        ).to(device)
        
        # Build a synthetic batch
        batch_size = 4
        seq_len = 11
        test_images = torch.randn(batch_size, seq_len, 3, 224, 224).to(device)
        
        print(f"\nInput shape: {test_images.shape}")
        
        with torch.no_grad():
            logits = model(test_images)
        
        print(f"Output logits shape: {logits.shape}")
        print(f"Predicted classes: {torch.argmax(logits, dim=1).tolist()}")
        
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Model info
        info = model.get_model_info()
        print(f"\nModel info: {info}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)