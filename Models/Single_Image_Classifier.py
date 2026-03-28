import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights


class AlexNetEncoder(nn.Module):
    """AlexNet - info"""
    
    def __init__(self, input_channels=3, use_pretrain=True):
        super().__init__()
        
        # AlexNet
        if use_pretrain:
            self.alexnet = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
            print("AlexNet: ImageNet pretrained weights")
        else:
            self.alexnet = alexnet(weights=None)
            print("AlexNet: info")
        
        # AlexNet feature extractor
        self.features = self.alexnet.features
        
        # Step 3ïinfo
        if input_channels != 3:
            old_conv = self.features[0]
            self.features[0] = nn.Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding
            )
        
        # AlexNetStep 256
        self.feature_dim = 256
        
        # info
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        """
        Args:
            x: [batch, channels, H, W]
        Returns:
            features: [batch, 256] info
        """
        # AlexNet feature extractor
        features = self.features(x)
        
        # info
        features = self.global_pool(features)
        features = features.flatten(1)
        
        return features


class SingleImageClassifier(nn.Module):
    """info"""
    
    def __init__(self, 
                 use_pretrain=True,
                 input_channels=3,
                 hidden_dim=128,
                 dropout=0.1,
                 num_classes=11):
        super().__init__()
        
        self.use_pretrain = use_pretrain
        self.num_classes = num_classes
        
        # info - AlexNet
        self.visual_encoder = AlexNetEncoder(
            input_channels=input_channels,
            use_pretrain=use_pretrain
        )
        
        # info
        feature_dim = self.visual_encoder.feature_dim  # 256
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        print(f"SingleImageClassifier info:")
        print(f"  AlexNet: {use_pretrain}")
        print(f"  info: {feature_dim}")
        print(f"  info: {hidden_dim}")
        print(f"  info: {num_classes}")
    
    def forward(self, x):
        """
        info
        
        Args:
            x: [batch, channels, H, W]
        
        Returns:
            logits: [batch, num_classes]
        """
        # info
        features = self.visual_encoder(x)
        
        # info
        logits = self.classifier(features)
        
        return logits
    
    def get_model_info(self):
        """info"""
        return {
            'model_type': 'SingleImageClassifier',
            'visual_encoder': 'AlexNet',
            'pretrained': self.use_pretrain,
            'num_classes': self.num_classes,
            'tasks': ['classification']
        }


def create_single_image_model(num_classes=11, use_pretrain=True, input_channels=3):
    """
    info
    
    Args:
        num_classes: info
        use_pretrain: AlexNet
        input_channels: info
    """
    model = SingleImageClassifier(
        use_pretrain=use_pretrain,
        input_channels=input_channels,
        hidden_dim=128,
        dropout=0.1,
        num_classes=num_classes
    )
    
    pretrain_str = "info" if use_pretrain else "info"
    print(f"info - AlexNet ({pretrain_str})")
    
    return model


# info
if __name__ == "__main__":
    print("=== info ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  # force CPU for testing
    print(f"info: {device}\n")
    
    # info
    test_configs = [
        (False, "info"),
        (True, "info"),
    ]
    
    for use_pretrain, desc in test_configs:
        print(f"\n{'='*60}")
        print(f"info: {desc}")
        print('='*60)
        
        model = create_single_image_model(
            num_classes=11,
            use_pretrain=use_pretrain,
            input_channels=3
        ).to(device)
        
        # info
        batch_size = 4
        test_images = torch.randn(batch_size, 3, 224, 224).to(device)
        
        print(f"\n: {test_images.shape}")
        
        with torch.no_grad():
            logits = model(test_images)
        
        print(f"info logits info: {logits.shape}")
        print(f"info: {torch.argmax(logits, dim=1).tolist()}")
        
        # info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n: {total_params:,}")
        print(f"info: {trainable_params:,}")
        
        # info
        info = model.get_model_info()
        print(f"\n: {info}")
    
    print("\n" + "="*60)
    print("info!")
    print("="*60)
