import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PerceptualLoss(nn.Module):
    """
    Perceptual Loss - Uses a pre-trained VGG network to extract features.
    Device-compatible version.
    """
    def __init__(self, layers=['relu1_1', 'relu2_1', 'relu3_1'], weights=[1.0, 1.0, 1.0], device='cuda:0'):
        super(PerceptualLoss, self).__init__()
        
        self.device = device
        
        # Manually create VGG feature extractor
        self.features = nn.ModuleList()
        
        # VGG19 early layers architecture (up to relu3_1)
        # Conv1
        self.features.append(nn.Conv2d(3, 64, kernel_size=3, padding=1))
        self.features.append(nn.ReLU(inplace=True))  # relu1_1
        self.features.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Conv2  
        self.features.append(nn.Conv2d(64, 128, kernel_size=3, padding=1))
        self.features.append(nn.ReLU(inplace=True))  # relu2_1
        self.features.append(nn.Conv2d(128, 128, kernel_size=3, padding=1))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Conv3
        self.features.append(nn.Conv2d(128, 256, kernel_size=3, padding=1))
        self.features.append(nn.ReLU(inplace=True))  # relu3_1
        
        # Layer indices for feature extraction
        self.extract_indices = [1, 6, 11]  # relu1_1, relu2_1, relu3_1
        self.weights = weights
        
        # Load pre-trained weights
        self._load_pretrained_weights()
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Normalization parameters
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Move to the specified device
        self.to(device)
        
    def _load_pretrained_weights(self):
        """Load pre-trained VGG19 weights"""
        try:
            vgg19 = models.vgg19(pretrained=True)
            vgg_features = vgg19.features
            
            # Map weights to our layers
            layer_mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            
            for i, vgg_idx in enumerate(layer_mapping):
                if i < len(self.features) and vgg_idx < len(vgg_features):
                    if isinstance(self.features[i], nn.Conv2d) and isinstance(vgg_features[vgg_idx], nn.Conv2d):
                        self.features[i].weight.data = vgg_features[vgg_idx].weight.data.clone()
                        self.features[i].bias.data = vgg_features[vgg_idx].bias.data.clone()
        except:
            print("WARNING: Unable to load pre-trained weights. Using random initialization.")
            
    def normalize_for_vgg(self, x):
        """Normalize input for VGG"""
        x = torch.clamp(x, 0, 1)
        x = (x - self.mean) / self.std
        return x
        
    def forward(self, pred, target):
        """Calculate perceptual loss"""
        # Ensure inputs are on the correct device
        pred = pred.to(self.device)
        target = target.to(self.device)
        
        # Normalize
        pred = self.normalize_for_vgg(pred)
        target = self.normalize_for_vgg(target)
        
        # Extract features
        pred_features = []
        target_features = []
        
        pred_x = pred
        target_x = target
        
        for i, layer in enumerate(self.features):
            pred_x = layer(pred_x)
            target_x = layer(target_x)
            
            if i in self.extract_indices:
                pred_features.append(pred_x)
                target_features.append(target_x)
        
        # Calculate loss
        total_loss = 0
        for i, (pred_feat, target_feat) in enumerate(zip(pred_features, target_features)):
            loss = F.mse_loss(pred_feat, target_feat)
            weight = self.weights[i] if i < len(self.weights) else 1.0
            total_loss += weight * loss
        
        return total_loss / len(pred_features)

class UnderwaterLoss(nn.Module):
    """
    Enhanced Multi-Task Loss Function - Includes Perceptual Loss
    """
    def __init__(self, depth_weight=0.0, denoise_weight=0.25, enhance_weight=0.5, perceptual_weight=0.25, device='cuda:0'):
        super(UnderwaterLoss, self).__init__()
        self.depth_weight = depth_weight
        self.denoise_weight = denoise_weight
        self.enhance_weight = enhance_weight
        self.perceptual_weight = perceptual_weight
        
        # Basic loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Perceptual loss - explicitly specifying the device
        self.perceptual_loss = PerceptualLoss(
            layers=['relu1_1', 'relu2_1', 'relu3_1'],
            weights=[1.0, 1.0, 1.0],
            device=device
        )
        
    def forward(self, outputs, targets):
        """
        Calculate total loss
        Args:
            outputs: Dictionary containing network outputs
            targets: Dictionary containing target data ('clean_image', 'depth_gt' optional)
        Returns:
            total_loss: The combined scalar loss
            losses: Dictionary containing individual loss components
        """
        total_loss = 0
        losses = {}
        
        # Main enhancement loss (L1)
        enhance_loss = self.l1_loss(outputs['enhanced_image'], targets['clean_image'])
        losses['enhance_loss'] = enhance_loss
        total_loss += self.enhance_weight * enhance_loss
        
        # Denoising loss (MSE)
        denoise_loss = self.mse_loss(outputs['denoised_image'], targets['clean_image'])
        losses['denoise_loss'] = denoise_loss
        total_loss += self.denoise_weight * denoise_loss
        
        # Perceptual loss - improves visual quality
        perceptual_loss = self.perceptual_loss(outputs['enhanced_image'], targets['clean_image'])
        losses['perceptual_loss'] = perceptual_loss
        total_loss += self.perceptual_weight * perceptual_loss
        
        # Depth loss (if ground truth is provided)
        if 'depth_gt' in targets:
            depth_loss = self.l1_loss(outputs['depth_map'], targets['depth_gt'])
            losses['depth_loss'] = depth_loss
            total_loss += self.depth_weight * depth_loss
        
        losses['total_loss'] = total_loss
        return total_loss, losses
