import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights


class AlexNetEncoder(nn.Module):
    """AlexNet视觉编码器 - 只提取最终特征"""
    
    def __init__(self, input_channels=3, use_pretrain=True):
        super().__init__()
        
        # 加载AlexNet
        if use_pretrain:
            self.alexnet = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
            print("AlexNet: 使用ImageNet预训练权重")
        else:
            self.alexnet = alexnet(weights=None)
            print("AlexNet: 随机初始化")
        
        # 提取AlexNet的特征层
        self.features = self.alexnet.features
        
        # 如果输入不是3通道，修改第一层
        if input_channels != 3:
            old_conv = self.features[0]
            self.features[0] = nn.Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding
            )
        
        # AlexNet最终特征维度为256
        self.feature_dim = 256
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        """
        Args:
            x: [batch, channels, H, W]
        Returns:
            features: [batch, 256] 全局池化后的特征
        """
        # 通过AlexNet特征提取层
        features = self.features(x)
        
        # 全局平均池化
        features = self.global_pool(features)
        features = features.flatten(1)
        
        return features


class SequencePoolingClassifier(nn.Module):
    """序列池化分类模型 - 无时序建模，只做特征聚合"""
    
    def __init__(self, 
                 use_pretrain=True,
                 input_channels=3,
                 pooling_strategy='mean',
                 hidden_dim=128,
                 dropout=0.1,
                 num_classes=11):
        """
        初始化序列池化分类器
        
        Args:
            use_pretrain: 是否使用预训练的AlexNet
            input_channels: 输入图像通道数
            pooling_strategy: 池化策略 ('mean', 'max', 'last')
            hidden_dim: 分类头隐藏层维度
            dropout: Dropout比率
            num_classes: 分类类别数
        """
        super().__init__()
        
        self.use_pretrain = use_pretrain
        self.pooling_strategy = pooling_strategy
        self.num_classes = num_classes
        
        # 验证池化策略
        assert pooling_strategy in ['mean', 'max', 'last'], \
            f"pooling_strategy must be 'mean', 'max', or 'last', got {pooling_strategy}"
        
        # 视觉编码器 - AlexNet（共享权重）
        self.visual_encoder = AlexNetEncoder(
            input_channels=input_channels,
            use_pretrain=use_pretrain
        )
        
        # 分类头
        feature_dim = self.visual_encoder.feature_dim  # 256
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        print(f"SequencePoolingClassifier 初始化:")
        print(f"  AlexNet预训练: {use_pretrain}")
        print(f"  池化策略: {pooling_strategy}")
        print(f"  特征维度: {feature_dim}")
        print(f"  隐藏层维度: {hidden_dim}")
        print(f"  分类数: {num_classes}")
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch, seq_len, channels, H, W]
        
        Returns:
            logits: [batch, num_classes]
        """
        batch_size, seq_len = x.shape[:2]
        
        # Reshape: [B, S, C, H, W] -> [B*S, C, H, W]
        x_flat = x.view(batch_size * seq_len, *x.shape[2:])
        
        # 提取视觉特征: [B*S, 256]
        features_flat = self.visual_encoder(x_flat)
        
        # Reshape回序列: [B*S, 256] -> [B, S, 256]
        features = features_flat.view(batch_size, seq_len, -1)
        
        # 池化聚合
        if self.pooling_strategy == 'mean':
            # 平均池化
            pooled_features = torch.mean(features, dim=1)  # [B, 256]
        elif self.pooling_strategy == 'max':
            # 最大池化
            pooled_features = torch.max(features, dim=1)[0]  # [B, 256]
        elif self.pooling_strategy == 'last':
            # 使用最后一帧
            pooled_features = features[:, -1, :]  # [B, 256]
        else:
            raise ValueError(f"Unknown pooling_strategy: {self.pooling_strategy}")
        
        # 分类
        logits = self.classifier(pooled_features)
        
        return logits
    
    def get_model_info(self):
        """获取模型信息"""
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
    创建序列池化分类模型的工厂函数
    
    Args:
        num_classes: 分类类别数
        use_pretrain: 是否使用预训练的AlexNet
        input_channels: 输入图像通道数
        pooling_strategy: 池化策略 ('mean', 'max', 'last')
    """
    model = SequencePoolingClassifier(
        use_pretrain=use_pretrain,
        input_channels=input_channels,
        pooling_strategy=pooling_strategy,
        hidden_dim=128,
        dropout=0.1,
        num_classes=num_classes
    )
    
    pretrain_str = "预训练" if use_pretrain else "随机初始化"
    print(f"创建序列池化分类模型 - AlexNet ({pretrain_str}) - {pooling_strategy} pooling")
    
    return model


# 测试代码
if __name__ == "__main__":
    print("=== 序列池化分类模型测试 ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  # 强制使用CPU进行测试
    print(f"使用设备: {device}\n")
    
    # 测试不同配置
    test_configs = [
        (False, 'mean', "随机初始化 + 平均池化"),
        (True, 'mean', "预训练 + 平均池化"),
        (True, 'max', "预训练 + 最大池化"),
        (True, 'last', "预训练 + 最后帧"),
    ]
    
    for use_pretrain, pool_strategy, desc in test_configs:
        print(f"\n{'='*60}")
        print(f"测试: {desc}")
        print('='*60)
        
        model = create_sequence_pooling_model(
            num_classes=11,
            use_pretrain=use_pretrain,
            input_channels=3,
            pooling_strategy=pool_strategy
        ).to(device)
        
        # 测试前向传播
        batch_size = 4
        seq_len = 11
        test_images = torch.randn(batch_size, seq_len, 3, 224, 224).to(device)
        
        print(f"\n测试输入形状: {test_images.shape}")
        
        with torch.no_grad():
            logits = model(test_images)
        
        print(f"输出 logits 形状: {logits.shape}")
        print(f"预测类别: {torch.argmax(logits, dim=1).tolist()}")
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
        # 获取模型信息
        info = model.get_model_info()
        print(f"\n模型信息: {info}")
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
