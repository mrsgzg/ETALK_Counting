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


class SingleImageClassifier(nn.Module):
    """Single-image classifier — encodes one frame at a time with no temporal modelling."""
    
    def __init__(self, 
                 use_pretrain=True,
                 input_channels=3,
                 hidden_dim=128,
                 dropout=0.1,
                 num_classes=11):
        super().__init__()
        
        self.use_pretrain = use_pretrain
        self.num_classes = num_classes
        
        # Visual encoder — AlexNet backbone
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
        
        print(f"SingleImageClassifier configuration:")
        print(f"  AlexNet pretrained: {use_pretrain}")
        print(f"  Visual feature dim: {feature_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num classes: {num_classes}")
    
    def forward(self, x):
        """
        Forward pass for a single image.
        
        Args:
            x: [batch, channels, H, W]
        
        Returns:
            logits: [batch, num_classes]
        """
        # Extract visual features
        features = self.visual_encoder(x)
        
        # Classify the feature vector
        logits = self.classifier(features)
        
        return logits
    
    def get_model_info(self):
        """Return a dict summarising the model configuration."""
        return {
            'model_type': 'SingleImageClassifier',
            'visual_encoder': 'AlexNet',
            'pretrained': self.use_pretrain,
            'num_classes': self.num_classes,
            'tasks': ['classification']
        }


def create_single_image_model(num_classes=11, use_pretrain=True, input_channels=3):
    """
    Factory function to instantiate a SingleImageClassifier.
    
    Args:
        num_classes: number of output classes
        use_pretrain: whether to load ImageNet pretrained weights for AlexNet
        input_channels: number of input image channels
    """
    model = SingleImageClassifier(
        use_pretrain=use_pretrain,
        input_channels=input_channels,
        hidden_dim=128,
        dropout=0.1,
        num_classes=num_classes
    )
    
    pretrain_str = "pretrained" if use_pretrain else "random init"
    print(f"Model created — AlexNet ({pretrain_str})")
    
    return model


# Usage example
if __name__ == "__main__":
    print("=== SingleImageClassifier Test ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  # force CPU for testing
    print(f"Device: {device}\n")
    
    # Test combinations: with and without pretrained weights
    test_configs = [
        (False, "no pretrain"),
        (True,  "pretrained"),
    ]
    
    for use_pretrain, desc in test_configs:
        print(f"\n{'='*60}")
        print(f"Configuration: {desc}")
        print('='*60)
        
        model = create_single_image_model(
            num_classes=11,
            use_pretrain=use_pretrain,
            input_channels=3
        ).to(device)
        
        # Build a synthetic batch
        batch_size = 4
        test_images = torch.randn(batch_size, 3, 224, 224).to(device)
        
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