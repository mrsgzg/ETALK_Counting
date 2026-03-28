import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import torchvision.models as models
from torchvision.models import AlexNet_Weights


class AlexNetEncoder(nn.Module):
    """简化的AlexNet视觉编码器 - 只提取最终特征"""
    
    def __init__(self, input_channels=3, use_pretrain=True):
        super().__init__()
        
        # 加载AlexNet
        if use_pretrain:
            self.alexnet = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
            print("使用预训练AlexNet权重 (weights=IMAGENET1K_V1)")
        else:
            self.alexnet = models.alexnet(weights=None)
            print("使用随机初始化AlexNet权重 (weights=None)")
        
        # 提取AlexNet的特征层
        self.features = self.alexnet.features
        
        # 如果输入不是3通道，修改第一层
        if input_channels != 3:
            original_conv1 = self.features[0]
            self.features[0] = nn.Conv2d(
                input_channels, 64, kernel_size=11, stride=4, padding=2
            )
            # 如果是预训练模型且输入通道为1，平均RGB权重
            if use_pretrain and input_channels == 1:
                with torch.no_grad():
                    self.features[0].weight = nn.Parameter(
                        original_conv1.weight.mean(dim=1, keepdim=True)
                    )
        
        # AlexNet最终特征维度为256
        self.feature_dim = 256
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 用于存储中间激活（可解释性）
        self.activations = {}
        self.register_hooks()
        
    def register_hooks(self):
        """注册前向钩子用于提取中间层激活"""
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # 注册关键层的钩子
        # Conv1 (index 0), Conv3 (index 6), Conv5 (index 10)
        self.features[0].register_forward_hook(get_activation('conv1'))
        self.features[6].register_forward_hook(get_activation('conv3'))
        self.features[10].register_forward_hook(get_activation('conv5'))
        
    def forward(self, x):
        """
        Args:
            x: [batch, channels, H, W]
        Returns:
            features: [batch, 256] 全局池化后的特征
        """
        # 通过AlexNet特征提取层
        features = self.features(x)
        
        # 保存最终卷积特征（用于可视化）
        self.activations['final_conv'] = features.detach()
        
        # 全局平均池化
        features = self.global_pool(features)
        features = features.flatten(1)
        
        return features
    
    def get_activations(self):
        """获取保存的激活值（用于可解释性分析）"""
        return self.activations
    
    def clear_activations(self):
        """清除保存的激活值"""
        self.activations.clear()


class JointEncoder(nn.Module):
    """关节编码器 - 简单MLP"""
    
    def __init__(self, joint_dim=2, hidden_dim=256, dropout=0.1):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, joint_positions):
        """
        Args:
            joint_positions: [batch, joint_dim] 只有joint1和joint6
        Returns:
            features: [batch, hidden_dim]
        """
        return self.encoder(joint_positions)


class CountingDecoder(nn.Module):
    """计数解码器"""
    
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=11):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        return self.decoder(x)


class JointDecoder(nn.Module):
    """关节预测解码器"""
    
    def __init__(self, input_dim=512, hidden_dim=256, joint_dim=2):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, joint_dim)
        )
        
    def forward(self, x):
        return self.decoder(x)


class ModalityGate(nn.Module):
    """模态门控机制 - 学习视觉和关节特征的重要性权重"""
    
    def __init__(self, visual_dim, joint_dim):
        super().__init__()
        
        # 计算gate权重
        total_dim = visual_dim + joint_dim
        self.gate_network = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(),
            nn.Linear(total_dim // 2, 2),  # 输出2个权重：visual和joint
            nn.Softmax(dim=-1)  # 归一化为概率分布
        )
        
        self.visual_dim = visual_dim
        self.joint_dim = joint_dim
        
    def forward(self, visual_features, joint_features):
        """
        Args:
            visual_features: [batch, visual_dim]
            joint_features: [batch, joint_dim]
        Returns:
            gated_visual: [batch, visual_dim] 加权后的视觉特征
            gated_joint: [batch, joint_dim] 加权后的关节特征
            gate_weights: [batch, 2] 权重 [visual_weight, joint_weight]
        """
        # 拼接特征用于计算gate
        combined = torch.cat([visual_features, joint_features], dim=1)
        
        # 计算gate权重
        gate_weights = self.gate_network(combined)  # [batch, 2]
        
        # 提取各模态权重
        visual_weight = gate_weights[:, 0:1]  # [batch, 1]
        joint_weight = gate_weights[:, 1:2]   # [batch, 1]
        
        # 加权特征
        gated_visual = visual_features * visual_weight
        gated_joint = joint_features * joint_weight
        
        return gated_visual, gated_joint, gate_weights


class SimplifiedEmbodiedCountingModel(nn.Module):
    """简化的具身计数模型（带模态门控）"""
    
    def __init__(self, 
                 use_pretrain=True,
                 lstm_layers=2,
                 lstm_hidden_size=512,
                 feature_dim=256,
                 joint_dim=2,  # 只有joint1和joint6
                 input_channels=3,
                 dropout=0.1,
                 num_classes=11,
                 use_modality_gate=True,  # 是否使用模态门控
                 **kwargs):
        super().__init__()
        
        self.lstm_layers = lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.joint_dim = joint_dim
        self.use_pretrain = use_pretrain
        self.use_modality_gate = use_modality_gate
        
        # 视觉编码器 - AlexNet
        self.visual_encoder = AlexNetEncoder(
            input_channels=input_channels,
            use_pretrain=use_pretrain
        )
        
        # 关节编码器 - MLP
        self.joint_encoder = JointEncoder(
            joint_dim=joint_dim,
            hidden_dim=feature_dim,
            dropout=dropout
        )
        
        # 模态门控机制（可选）
        visual_dim = self.visual_encoder.feature_dim  # 256
        joint_encoded_dim = feature_dim  # 256
        
        if use_modality_gate:
            self.modality_gate = ModalityGate(visual_dim, joint_encoded_dim)
            print(f"  使用模态门控机制: 是")
        else:
            self.modality_gate = None
            print(f"  使用模态门控机制: 否")
        
        # 特征融合：简单拼接
        lstm_input_dim = visual_dim + joint_encoded_dim  # 512
        
        # LSTM时序建模
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # 计数解码器
        self.counting_decoder = CountingDecoder(
            input_dim=lstm_hidden_size,
            hidden_dim=feature_dim,
            num_classes=num_classes
        )
        
        # 关节预测解码器
        self.joint_decoder = JointDecoder(
            input_dim=lstm_hidden_size,
            hidden_dim=feature_dim,
            joint_dim=joint_dim
        )
        
        # 用于存储LSTM隐藏状态（可解释性）
        self.lstm_hidden_states = []
        self.lstm_cell_states = []
        
        # 用于存储gate权重（可解释性）
        self.gate_weights_history = []
        
        print(f"SimplifiedEmbodiedCountingModel 初始化:")
        print(f"  AlexNet预训练: {use_pretrain}")
        print(f"  视觉特征维度: {visual_dim}")
        print(f"  关节特征维度: {joint_encoded_dim}")
        print(f"  LSTM输入维度: {lstm_input_dim}")
        print(f"  LSTM隐藏层大小: {lstm_hidden_size}")
        print(f"  关节维度: {joint_dim} (joint1 + joint6)")
        print(f"  模态门控: {use_modality_gate}")
        
    def init_lstm_hidden(self, batch_size, device):
        """初始化LSTM隐藏状态"""
        h0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        return (h0, c0)
    
    def forward(self, sequence_data, use_teacher_forcing=True, return_hidden_states=False):
        """
        前向传播
        
        Args:
            sequence_data: 字典包含
                - 'images': [batch, seq_len, channels, H, W]
                - 'joints': [batch, seq_len, 2]  只有joint1和joint6
            use_teacher_forcing: 是否使用teacher forcing
            return_hidden_states: 是否返回LSTM隐藏状态
        
        Returns:
            outputs: 字典包含
                - 'counts': [batch, seq_len, num_classes] 计数预测
                - 'joints': [batch, seq_len, 2] 关节预测
                - 'hidden_states': [batch, seq_len, hidden_dim] (可选)
        """
        images = sequence_data['images']
        joints = sequence_data['joints']
        
        batch_size, seq_len = images.shape[:2]
        device = images.device
        
        # 初始化LSTM隐藏状态
        lstm_hidden = self.init_lstm_hidden(batch_size, device)
        
        # 存储预测结果
        count_predictions = []
        joint_predictions = []
        
        # 清空历史记录
        self.lstm_hidden_states.clear()
        self.lstm_cell_states.clear()
        self.gate_weights_history.clear()
        self.visual_encoder.clear_activations()
        
        # 当前关节状态
        current_joints = joints[:, 0]
        
        for t in range(seq_len):
            # 1. 视觉特征提取 - AlexNet
            visual_features = self.visual_encoder(images[:, t])
            
            # 2. 关节特征编码 - MLP
            joint_features = self.joint_encoder(current_joints)
            
            # 3. 模态门控（如果启用）
            if self.use_modality_gate:
                visual_features, joint_features, gate_weights = self.modality_gate(
                    visual_features, joint_features
                )
                # 保存gate权重用于分析
                self.gate_weights_history.append(gate_weights.detach().clone())
            
            # 4. 特征融合 - 简单拼接
            combined_features = torch.cat([visual_features, joint_features], dim=1)
            
            # 5. LSTM时序建模
            lstm_input = combined_features.unsqueeze(1)  # [batch, 1, feature_dim]
            lstm_output, lstm_hidden = self.lstm(lstm_input, lstm_hidden)
            lstm_output = lstm_output.squeeze(1)  # [batch, hidden_dim]
            
            # 保存LSTM状态（可解释性）
            if return_hidden_states:
                self.lstm_hidden_states.append(lstm_hidden[0][-1].detach().clone())
                self.lstm_cell_states.append(lstm_hidden[1][-1].detach().clone())
            
            # 6. 计数预测
            count_pred = self.counting_decoder(lstm_output)
            count_predictions.append(count_pred)
            
            # 7. 关节预测
            joint_pred = self.joint_decoder(lstm_output)
            joint_predictions.append(joint_pred)
            
            # 8. 更新当前关节位置
            if use_teacher_forcing and t < seq_len - 1:
                current_joints = joints[:, t + 1]
            else:
                current_joints = joint_pred
        
        # 组织输出
        outputs = {
            'counts': torch.stack(count_predictions, dim=1),
            'joints': torch.stack(joint_predictions, dim=1)
        }
        
        if return_hidden_states:
            outputs['hidden_states'] = torch.stack(self.lstm_hidden_states, dim=1)
            outputs['cell_states'] = torch.stack(self.lstm_cell_states, dim=1)
        
        # 如果使用了gate，也返回gate权重
        if self.use_modality_gate and len(self.gate_weights_history) > 0:
            outputs['gate_weights'] = torch.stack(self.gate_weights_history, dim=1)
        
        return outputs
    
    def get_visual_activations(self):
        """获取视觉编码器的激活值（用于CAM等可视化）"""
        return self.visual_encoder.get_activations()
    
    def get_lstm_states(self):
        """获取LSTM隐藏状态（用于时序分析）"""
        return {
            'hidden_states': self.lstm_hidden_states,
            'cell_states': self.lstm_cell_states
        }
    
    def get_gate_weights(self):
        """获取模态门控权重（用于分析模态重要性）"""
        if self.use_modality_gate and len(self.gate_weights_history) > 0:
            return torch.stack(self.gate_weights_history, dim=0)  # [seq_len, batch, 2]
        return None
    
    def enable_grad_cam(self):
        """启用梯度计算用于Grad-CAM"""
        for param in self.visual_encoder.features.parameters():
            param.requires_grad = True
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_type': 'SimplifiedEmbodiedCountingModel',
            'visual_encoder': 'AlexNet',
            'pretrained': self.use_pretrain,
            'has_attention': False,
            'has_fovea_bias': False,
            'has_multi_scale': False,
            'fusion_type': 'concatenation',
            'modality_gate': self.use_modality_gate,
            'joint_dim': self.joint_dim,
            'tasks': ['counting', 'joint_prediction'],
            'interpretability_features': [
                'visual_activations',
                'lstm_hidden_states',
                'modality_gate_weights',
                'grad_cam_ready'
            ]
        }


def create_model(config, use_pretrain=True, use_modality_gate=True):
    """
    创建简化具身计数模型的工厂函数
    
    Args:
        config: 模型配置字典
        use_pretrain: 是否使用预训练的AlexNet
        use_modality_gate: 是否使用模态门控机制
    """
    image_mode = config.get('image_mode', 'rgb')
    input_channels = 3 if image_mode == 'rgb' else 1
    
    model_config = config['model_config'].copy()
    model_config['input_channels'] = input_channels
    model_config['use_pretrain'] = use_pretrain
    model_config['use_modality_gate'] = use_modality_gate
    
    model = SimplifiedEmbodiedCountingModel(**model_config)
    
    pretrain_str = "预训练" if use_pretrain else "随机初始化"
    gate_str = "带门控" if use_modality_gate else "无门控"
    print(f"创建简化具身计数模型 - AlexNet ({pretrain_str}) - {gate_str}")
    
    return model


# 测试代码
if __name__ == "__main__":
    print("=== 简化具身计数模型测试 ===\n")
    
    config = {
        'image_mode': 'rgb',
        'model_config': {
            'lstm_layers': 2,
            'lstm_hidden_size': 512,
            'feature_dim': 256,
            'joint_dim': 2,  # joint1 + joint6
            'dropout': 0.1,
            'num_classes': 11
        }
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  # 强制使用CPU进行测试
    print(f"使用设备: {device}\n")
    
    # 测试四个版本：预训练/非预训练 × 门控/非门控
    test_configs = [
        (False, False, "随机初始化 + 无门控"),
        (True,  True,  "预训练 + 带门控"),
    ]
    
    for use_pretrain, use_gate, desc in test_configs:
        print(f"\n{'='*60}")
        print(f"测试: {desc}")
        print('='*60)
        
        model = create_model(config, 
                           use_pretrain=use_pretrain,
                           use_modality_gate=use_gate).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n模型参数:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        
        # 测试数据
        batch_size = 4
        seq_len = 6
        sequence_data = {
            'images': torch.randn(batch_size, seq_len, 3, 224, 224, device=device),
            'joints': torch.randn(batch_size, seq_len, 2, device=device),  # 只有2个joint
        }
        
        # 前向传播测试
        model.eval()
        with torch.no_grad():
            outputs = model(sequence_data, return_hidden_states=True)
        
        print(f"\n输出形状:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        
        # 测试可解释性功能
        print(f"\n可解释性功能:")
        activations = model.get_visual_activations()
        print(f"  视觉激活层数: {len(activations)}")
        for name, act in activations.items():
            print(f"    {name}: {act.shape}")
        
        lstm_states = model.get_lstm_states()
        print(f"  LSTM隐藏状态: {len(lstm_states['hidden_states'])} 时间步")
        
        # 测试gate权重（如果有）
        if use_gate:
            gate_weights = model.get_gate_weights()
            if gate_weights is not None:
                print(f"  门控权重: {gate_weights.shape}")
                print(f"  平均门控权重 [visual, joint]: {gate_weights.mean(dim=[0,1]).tolist()}")
        
        print(f"\n模型信息:")
        info = model.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print(f"\n模型信息:")
        info = model.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 清理内存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)

