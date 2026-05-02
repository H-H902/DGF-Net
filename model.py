import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual Block - Used to mitigate the vanishing gradient problem.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Residual connection
        out = self.relu(out)
        return out

class DepthEstimator(nn.Module):
    """
    Depth Estimation Module - Simplified implementation with residual connections.
    """
    def __init__(self, input_channels=3, depth_channels=1):
        super(DepthAnythingModule, self).__init__()
        
        # Encoder - Extract multi-scale features (with residual blocks)
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, 64, 7, 2, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                ResidualBlock(64)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                ResidualBlock(128)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, 3, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                ResidualBlock(256)
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, 3, 2, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                ResidualBlock(512)
            )
        ])
        
        # Decoder - Generate depth map (with residual blocks)
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                ResidualBlock(256)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                ResidualBlock(128)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                ResidualBlock(64)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, depth_channels, 4, 2, 1),
                nn.Sigmoid()  # Normalize depth values to [0,1]
            )
        ])
        
    def forward(self, x):
        # Encoder forward
        features = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            features.append(x)
        
        # Decoder forward with skip connections
        x = features[-1]
        for i, decoder_layer in enumerate(self.decoder):
            x = decoder_layer(x)
            if i < len(self.decoder) - 1 and i < len(features) - 1:
                # Skip connection
                skip_feat = features[-(i+2)]
                if x.shape != skip_feat.shape:
                    skip_feat = F.interpolate(skip_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = x + skip_feat
        
        return x

class DepthGuidedDenoising(nn.Module):
    """
    Depth-Guided Denoising Module (with residual connections)
    """
    def __init__(self, rgb_channels=3, depth_channels=1, hidden_dim=64):
        super(DepthGuidedDenoising, self).__init__()
        
        # Depth feature extraction (2 layers)
        self.depth_feature_extractor = nn.Sequential(
            nn.Conv2d(depth_channels, hidden_dim//2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim//2, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        # RGB feature extraction (2 layers)
        self.rgb_feature_extractor = nn.Sequential(
            nn.Conv2d(rgb_channels, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        # Fusion network (3 layers, with residual block)
        self.fusion_network = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            ResidualBlock(hidden_dim),  # Added residual block
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, rgb_channels, 3, 1, 1),
            nn.Tanh()  # Output denoised RGB
        )
        
    def forward(self, rgb, depth):
        depth_feat = self.depth_feature_extractor(depth)
        rgb_feat = self.rgb_feature_extractor(rgb)
        
        # Feature fusion
        fused_feat = torch.cat([rgb_feat, depth_feat], dim=1)
        denoised_rgb = self.fusion_network(fused_feat)
        
        return denoised_rgb + rgb  # Residual connection

class FFTProcessor(nn.Module):
    """
    FFT Processing Module (with residual connections)
    """
    def __init__(self, channels=3):
        super(FFTProcessor, self).__init__()
        self.channels = channels
        
        # Frequency domain feature processing network (4 layers, with residual block)
        self.freq_processor = nn.Sequential(
            nn.Conv2d(channels * 2, 64, 3, 1, 1),  # *2 for real and imaginary parts
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            ResidualBlock(128),  # Added residual block
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, channels * 2, 3, 1, 1)
        )
        
    def forward(self, x):
        # RGB to FFT
        B, C, H, W = x.shape
        
        # Perform FFT on each channel separately
        fft_real_list = []
        fft_imag_list = []
        
        for c in range(C):
            channel = x[:, c:c+1, :, :]  # [B, 1, H, W]
            fft_channel = torch.fft.fft2(channel.squeeze(1))  # [B, H, W] complex
            fft_real_list.append(fft_channel.real.unsqueeze(1))
            fft_imag_list.append(fft_channel.imag.unsqueeze(1))
        
        fft_real = torch.cat(fft_real_list, dim=1)  # [B, C, H, W]
        fft_imag = torch.cat(fft_imag_list, dim=1)  # [B, C, H, W]
        
        # Combine real and imaginary parts
        fft_combined = torch.cat([fft_real, fft_imag], dim=1)  # [B, 2*C, H, W]
        
        # Frequency domain processing
        processed_fft = self.freq_processor(fft_combined)
        
        # Separate real and imaginary parts
        processed_real = processed_fft[:, :C, :, :]
        processed_imag = processed_fft[:, C:, :, :]
        
        # Inverse FFT
        enhanced_channels = []
        for c in range(C):
            real_part = processed_real[:, c, :, :]  # [B, H, W]
            imag_part = processed_imag[:, c, :, :]  # [B, H, W]
            complex_tensor = torch.complex(real_part, imag_part)
            ifft_channel = torch.fft.ifft2(complex_tensor).real  # [B, H, W]
            enhanced_channels.append(ifft_channel.unsqueeze(1))
        
        enhanced_rgb = torch.cat(enhanced_channels, dim=1)  # [B, C, H, W]
        
        return enhanced_rgb

class DGFNet(nn.Module):
    """
    Complete Underwater Image Enhancement Network (DGF-Net)
    Pipeline: Depth Estimation -> Depth-Guided Denoising -> FFT Enhancement -> Output
    """
    def __init__(self, input_channels=3, output_channels=3):
        super(DGFNet, self).__init__()
        
        # Stage 1: Depth Estimation
        self.depth_estimator = DepthEstimat0r(input_channels=input_channels)
        
        # Stage 2: Depth-Guided Denoising
        self.depth_guided_denoising = DepthGuidedDenoising(
            rgb_channels=input_channels, 
            depth_channels=1
        )
        
        # Stage 3: FFT Processing
        self.fft_processor = FFTProcessor(channels=input_channels)
        
        # Output adjustment layer (2 layers, with residual connection)
        self.output_adjust = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            ResidualBlock(64),  # Added residual block
            nn.Conv2d(64, output_channels, 3, 1, 1),
            nn.Sigmoid()  # Ensure output is in the [0,1] range
        )
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input underwater image [B, C, H, W]
        Returns:
            Dictionary containing:
            - enhanced_image: Enhanced image [B, C, H, W]
            - depth_map: Depth map [B, 1, H, W]
            - denoised_image: Denoised image [B, C, H, W]
            - fft_enhanced: FFT enhanced image [B, C, H, W]
        """
        # Stage 1: Depth Estimation
        depth_map = self.depth_estimator(x)
        
        # Stage 2: Depth-Guided Denoising
        denoised_image = self.depth_guided_denoising(x, depth_map)
        
        # Stage 3: FFT Enhancement
        fft_enhanced = self.fft_processor(denoised_image)
        
        # Final output adjustment
        enhanced_image = self.output_adjust(fft_enhanced)
        
        return {
            'enhanced_image': enhanced_image,
            'depth_map': depth_map,
            'denoised_image': denoised_image,
            'fft_enhanced': fft_enhanced
        }
