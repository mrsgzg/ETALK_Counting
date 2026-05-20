import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import torchvision.models as models
from torchvision.models import AlexNet_Weights


class AlexNetEncoder(nn.Module):
    """AlexNet-based visual feature extractor."""
    
    def __init__(self, input_channels=3, use_pretrain=True):
        super().__init__()
        
        # Load AlexNet with or without pretrained weights
        if use_pretrain:
            self.alexnet = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
            print("AlexNet feature extractor (weights=IMAGENET1K_V1)")
        else:
            self.alexnet = models.alexnet(weights=None)
            print("AlexNet feature extractor (weights=None)")
        
        # Use only the convolutional feature extractor
        self.features = self.alexnet.features
        
        # Adapt the first conv layer if input channels differ from 3
        if input_channels != 3:
            original_conv1 = self.features[0]
            self.features[0] = nn.Conv2d(
                input_channels, 64, kernel_size=11, stride=4, padding=2
            )
            # For grayscale input, initialise weights by averaging the RGB channels
            if use_pretrain and input_channels == 1:
                with torch.no_grad():
                    self.features[0].weight = nn.Parameter(
                        original_conv1.weight.mean(dim=1, keepdim=True)
                    )
        
        # AlexNet conv5 outputs 256 feature maps
        self.feature_dim = 256
        
        # Global average pooling to collapse spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Storage for intermediate activations (populated by forward hooks)
        self.activations = {}
        self.register_hooks()
        
    def register_hooks(self):
        """Register forward hooks to capture intermediate conv activations."""
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Hook Conv1 (index 0), Conv3 (index 6), Conv5 (index 10)
        self.features[0].register_forward_hook(get_activation('conv1'))
        self.features[6].register_forward_hook(get_activation('conv3'))
        self.features[10].register_forward_hook(get_activation('conv5'))
        
    def forward(self, x):
        """
        Args:
            x: [batch, channels, H, W]
        Returns:
            features: [batch, 256] global-pooled feature vector
        """
        # Pass through AlexNet convolutional layers
        features = self.features(x)
        
        # Store the final conv output before pooling
        self.activations['final_conv'] = features.detach()
        
        # Global average pool and flatten
        features = self.global_pool(features)
        features = features.flatten(1)
        
        return features
    
    def get_activations(self):
        """Return the dict of stored intermediate activations."""
        return self.activations
    
    def clear_activations(self):
        """Clear all stored activations."""
        self.activations.clear()


class JointEncoder(nn.Module):
    """Joint position encoder — MLP-based."""
    
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
            joint_positions: [batch, joint_dim]  — joint1 and joint6 angles
        Returns:
            features: [batch, hidden_dim]
        """
        return self.encoder(joint_positions)


class CountingDecoder(nn.Module):
    """Counting head — maps LSTM output to ball-count class logits."""
    
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
    """Joint prediction head — maps LSTM output to predicted joint positions."""
    
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
    """Modality gate — learns soft weights for visual and joint features."""
    
    def __init__(self, visual_dim, joint_dim):
        super().__init__()
        
        # Gate network: takes concatenated features and outputs per-modality weights
        total_dim = visual_dim + joint_dim
        self.gate_network = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(),
            nn.Linear(total_dim // 2, 2),  # 2 outputs: visual weight and joint weight
            nn.Softmax(dim=-1)             # weights sum to 1
        )
        
        self.visual_dim = visual_dim
        self.joint_dim = joint_dim
        
    def forward(self, visual_features, joint_features):
        """
        Args:
            visual_features: [batch, visual_dim]
            joint_features: [batch, joint_dim]
        Returns:
            gated_visual: [batch, visual_dim]  — scaled visual features
            gated_joint: [batch, joint_dim]    — scaled joint features
            gate_weights: [batch, 2]           — [visual_weight, joint_weight]
        """
        # Concatenate both modalities as gate input
        combined = torch.cat([visual_features, joint_features], dim=1)
        
        # Compute soft gate weights
        gate_weights = self.gate_network(combined)  # [batch, 2]
        
        # Split into per-modality scalars
        visual_weight = gate_weights[:, 0:1]  # [batch, 1]
        joint_weight = gate_weights[:, 1:2]   # [batch, 1]
        
        # Apply gates
        gated_visual = visual_features * visual_weight
        gated_joint = joint_features * joint_weight
        
        return gated_visual, gated_joint, gate_weights


class SimplifiedEmbodiedCountingModel(nn.Module):
    """Embodied counting model — AlexNet visual encoder + joint encoder + LSTM."""
    
    def __init__(self, 
                 use_pretrain=True,
                 lstm_layers=2,
                 lstm_hidden_size=512,
                 feature_dim=256,
                 joint_dim=2,          # joint1 and joint6
                 input_channels=3,
                 dropout=0.1,
                 num_classes=11,
                 use_modality_gate=True,  # enable/disable the modality gate
                 **kwargs):
        super().__init__()
        
        self.lstm_layers = lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.joint_dim = joint_dim
        self.use_pretrain = use_pretrain
        self.use_modality_gate = use_modality_gate
        
        # Visual encoder — AlexNet backbone
        self.visual_encoder = AlexNetEncoder(
            input_channels=input_channels,
            use_pretrain=use_pretrain
        )
        
        # Joint encoder — MLP
        self.joint_encoder = JointEncoder(
            joint_dim=joint_dim,
            hidden_dim=feature_dim,
            dropout=dropout
        )
        
        # Feature dimensions after encoding
        visual_dim = self.visual_encoder.feature_dim  # 256
        joint_encoded_dim = feature_dim               # 256
        
        if use_modality_gate:
            self.modality_gate = ModalityGate(visual_dim, joint_encoded_dim)
            print(f"  Modality gate: enabled")
        else:
            self.modality_gate = None
            print(f"  Modality gate: disabled")
        
        # LSTM input is the concatenation of visual and joint features
        lstm_input_dim = visual_dim + joint_encoded_dim  # 512
        
        # Temporal LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Ball-count classification head
        self.counting_decoder = CountingDecoder(
            input_dim=lstm_hidden_size,
            hidden_dim=feature_dim,
            num_classes=num_classes
        )
        
        # Joint position prediction head
        self.joint_decoder = JointDecoder(
            input_dim=lstm_hidden_size,
            hidden_dim=feature_dim,
            joint_dim=joint_dim
        )
        
        # Buffers for LSTM hidden and cell states (populated during forward pass)
        self.lstm_hidden_states = []
        self.lstm_cell_states = []
        
        # Buffer for gate weight history (populated during forward pass)
        self.gate_weights_history = []
        
        print(f"SimplifiedEmbodiedCountingModel configuration:")
        print(f"  AlexNet pretrained: {use_pretrain}")
        print(f"  Visual feature dim: {visual_dim}")
        print(f"  Joint feature dim: {joint_encoded_dim}")
        print(f"  LSTM input dim: {lstm_input_dim}")
        print(f"  LSTM hidden size: {lstm_hidden_size}")
        print(f"  Joint inputs: {joint_dim} (joint1 + joint6)")
        print(f"  Modality gate: {use_modality_gate}")
        
    def init_lstm_hidden(self, batch_size, device):
        """Initialise LSTM hidden and cell states to zeros."""
        h0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        return (h0, c0)
    
    def forward(self, sequence_data, use_teacher_forcing=True, return_hidden_states=False):
        """
        Forward pass over a full sequence.
        
        Args:
            sequence_data: dict containing:
                - 'images': [batch, seq_len, channels, H, W]
                - 'joints': [batch, seq_len, 2]  — joint1 and joint6
            use_teacher_forcing: if True, use ground-truth joints as next-step input
            return_hidden_states: if True, include LSTM states in output dict
        
        Returns:
            outputs: dict containing:
                - 'counts': [batch, seq_len, num_classes]  — per-frame count logits
                - 'joints': [batch, seq_len, 2]            — per-frame joint predictions
                - 'hidden_states': [batch, seq_len, hidden_dim]  (optional)
        """
        images = sequence_data['images']
        joints = sequence_data['joints']
        
        batch_size, seq_len = images.shape[:2]
        device = images.device
        
        # Initialise LSTM state
        lstm_hidden = self.init_lstm_hidden(batch_size, device)
        
        # Output accumulators
        count_predictions = []
        joint_predictions = []
        
        # Clear state buffers from the previous forward pass
        self.lstm_hidden_states.clear()
        self.lstm_cell_states.clear()
        self.gate_weights_history.clear()
        self.visual_encoder.clear_activations()
        
        # Use the first ground-truth joint as the initial joint input
        current_joints = joints[:, 0]
        
        for t in range(seq_len):
            # 1. Extract visual features with AlexNet
            visual_features = self.visual_encoder(images[:, t])
            
            # 2. Encode joint positions with MLP
            joint_features = self.joint_encoder(current_joints)
            
            # 3. Optionally apply the modality gate
            if self.use_modality_gate:
                visual_features, joint_features, gate_weights = self.modality_gate(
                    visual_features, joint_features
                )
                # Store gate weights for later analysis
                self.gate_weights_history.append(gate_weights.detach().clone())
            
            # 4. Fuse modalities by concatenation
            combined_features = torch.cat([visual_features, joint_features], dim=1)
            
            # 5. Step through LSTM
            lstm_input = combined_features.unsqueeze(1)  # [batch, 1, feature_dim]
            lstm_output, lstm_hidden = self.lstm(lstm_input, lstm_hidden)
            lstm_output = lstm_output.squeeze(1)         # [batch, hidden_dim]
            
            # Optionally store LSTM states
            if return_hidden_states:
                self.lstm_hidden_states.append(lstm_hidden[0][-1].detach().clone())
                self.lstm_cell_states.append(lstm_hidden[1][-1].detach().clone())
            
            # 6. Predict ball count
            count_pred = self.counting_decoder(lstm_output)
            count_predictions.append(count_pred)
            
            # 7. Predict next joint positions
            joint_pred = self.joint_decoder(lstm_output)
            joint_predictions.append(joint_pred)
            
            # 8. Decide next joint input: ground truth (teacher forcing) or prediction
            if use_teacher_forcing and t < seq_len - 1:
                current_joints = joints[:, t + 1]
            else:
                current_joints = joint_pred
        
        # Assemble output dict
        outputs = {
            'counts': torch.stack(count_predictions, dim=1),
            'joints': torch.stack(joint_predictions, dim=1)
        }
        
        if return_hidden_states:
            outputs['hidden_states'] = torch.stack(self.lstm_hidden_states, dim=1)
            outputs['cell_states'] = torch.stack(self.lstm_cell_states, dim=1)
        
        # Include gate weights if available
        if self.use_modality_gate and len(self.gate_weights_history) > 0:
            outputs['gate_weights'] = torch.stack(self.gate_weights_history, dim=1)
        
        return outputs
    
    def get_visual_activations(self):
        """Return intermediate conv activations (useful for CAM visualisation)."""
        return self.visual_encoder.get_activations()
    
    def get_lstm_states(self):
        """Return stored LSTM hidden and cell states from the last forward pass."""
        return {
            'hidden_states': self.lstm_hidden_states,
            'cell_states': self.lstm_cell_states
        }
    
    def get_gate_weights(self):
        """Return stacked gate weights from the last forward pass."""
        if self.use_modality_gate and len(self.gate_weights_history) > 0:
            return torch.stack(self.gate_weights_history, dim=0)  # [seq_len, batch, 2]
        return None
    
    def enable_grad_cam(self):
        """Ensure gradients flow through the visual encoder for Grad-CAM."""
        for param in self.visual_encoder.features.parameters():
            param.requires_grad = True
    
    def get_model_info(self):
        """Return a dict summarising the model configuration."""
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
    Factory function to instantiate SimplifiedEmbodiedCountingModel.
    
    Args:
        config: experiment config dict
        use_pretrain: whether to load ImageNet pretrained weights for AlexNet
        use_modality_gate: whether to enable the modality gate
    """
    image_mode = config.get('image_mode', 'rgb')
    input_channels = 3 if image_mode == 'rgb' else 1
    
    model_config = config['model_config'].copy()
    model_config['input_channels'] = input_channels
    model_config['use_pretrain'] = use_pretrain
    model_config['use_modality_gate'] = use_modality_gate
    
    model = SimplifiedEmbodiedCountingModel(**model_config)
    
    pretrain_str = "pretrained" if use_pretrain else "random init"
    gate_str = "gate enabled" if use_modality_gate else "gate disabled"
    print(f"Model created — AlexNet ({pretrain_str}) — {gate_str}")
    
    return model


# Usage example
if __name__ == "__main__":
    print("=== SimplifiedEmbodiedCountingModel Test ===\n")
    
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
    device = torch.device('cpu')  # force CPU for testing
    print(f"Device: {device}\n")
    
    # Test combinations: no pretrain / no gate  vs  pretrain / gate
    test_configs = [
        (False, False, "no pretrain + no gate"),
        (True,  True,  "pretrained + gate"),
    ]
    
    for use_pretrain, use_gate, desc in test_configs:
        print(f"\n{'='*60}")
        print(f"Configuration: {desc}")
        print('='*60)
        
        model = create_model(config, 
                           use_pretrain=use_pretrain,
                           use_modality_gate=use_gate).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nParameter count:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        
        # Build a synthetic batch
        batch_size = 4
        seq_len = 6
        sequence_data = {
            'images': torch.randn(batch_size, seq_len, 3, 224, 224, device=device),
            'joints': torch.randn(batch_size, seq_len, 2, device=device),  # 2 joint angles
        }
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(sequence_data, return_hidden_states=True)
        
        print(f"\nOutput shapes:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        
        # Inspect stored activations
        print(f"\nInterpretability outputs:")
        activations = model.get_visual_activations()
        print(f"  Conv activation layers: {len(activations)}")
        for name, act in activations.items():
            print(f"    {name}: {act.shape}")
        
        lstm_states = model.get_lstm_states()
        print(f"  LSTM hidden states: {len(lstm_states['hidden_states'])} timesteps")
        
        # Gate weight statistics
        if use_gate:
            gate_weights = model.get_gate_weights()
            if gate_weights is not None:
                print(f"  Gate weights shape: {gate_weights.shape}")
                print(f"  Mean gate weights [visual, joint]: {gate_weights.mean(dim=[0,1]).tolist()}")
        
        print(f"\nModel info:")
        info = model.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)