"""
Multi-modal neural network architectures for satellite imagery processing.
Implements various fusion strategies for combining SAR and optical data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50


class BimodalResNetClassifier(nn.Module):
    """
    Dual-pathway ResNet architecture for processing two distinct data modalities.
    Each modality is processed through its own ResNet backbone before fusion.
    
    Args:
        modality1_channels: Input channel count for first modality
        modality2_channels: Input channel count for second modality  
        output_classes: Number of target classification categories (default: 8)
        backbone_name: Name of ResNet variant ('resnet18' or 'resnet50', default: 'resnet50')
        feature_size: Dimensionality of concatenated feature vectors (default: 4096 for ResNet50)
    """
    
    def __init__(self, modality1_channels, modality2_channels, 
                 output_classes=8, backbone_name="resnet50", feature_size=4096):
        super().__init__()
        
        # Validate backbone name to prevent security issues
        valid_backbones = {'resnet18': resnet18, 'resnet50': resnet50}
        if backbone_name not in valid_backbones:
            raise ValueError(
                f"Invalid backbone_name: {backbone_name}. "
                f"Must be one of {list(valid_backbones.keys())}"
            )
        
        # Initialize backbone for first modality (e.g., SAR)
        # Use dictionary lookup instead of eval() for security
        backbone_fn = valid_backbones[backbone_name]
        self.pathway1 = backbone_fn(pretrained=False)
        # Modify first conv layer to accept modality1_channels instead of 3 (RGB)
        self.pathway1.conv1 = nn.Conv2d(
            modality1_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Remove classification head to use as feature extractor
        self.pathway1.fc = nn.Identity()
        
        # Initialize backbone for second modality (e.g., optical)
        self.pathway2 = backbone_fn(pretrained=False)
        # Modify first conv layer to accept modality2_channels instead of 3 (RGB)
        self.pathway2.conv1 = nn.Conv2d(
            modality2_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Remove classification head to use as feature extractor
        self.pathway2.fc = nn.Identity()
        
        # Final classification layer
        self.classifier_head = nn.Linear(feature_size, output_classes)
    
    def forward(self, input_dict):
        """
        Process multi-modal input through dual pathways.
        
        Args:
            input_dict: Dictionary with keys 's1' and 's2' containing tensor data
                - 's1': Tensor of shape (batch, modality1_channels, H, W)
                - 's2': Tensor of shape (batch, modality2_channels, H, W)
            
        Returns:
            torch.Tensor: Classification logits with shape (batch, output_classes)
            
        Raises:
            KeyError: If required keys ('s1', 's2') are missing from input_dict
        """
        # Extract input tensors from dictionary
        if "s1" not in input_dict or "s2" not in input_dict:
            raise KeyError("input_dict must contain both 's1' and 's2' keys")
        
        modality1_data = input_dict["s1"]
        modality2_data = input_dict["s2"]
        
        # Extract features from each pathway independently
        # Each pathway processes its modality through ResNet backbone
        features1 = self.pathway1(modality1_data)  # Shape: (batch, 2048)
        features2 = self.pathway2(modality2_data)  # Shape: (batch, 2048)
        
        # Concatenate features along channel dimension for late fusion
        # Result: (batch, 4096) for ResNet50 (2048 + 2048)
        combined_features = torch.cat([features1, features2], dim=1)
        
        # Final classification layer produces logits for each class
        predictions = self.classifier_head(combined_features)
        
        return predictions


class ConvolutionalBlock(nn.Module):
    """
    Standard convolutional building block with batch normalization and ReLU.
    Applies two consecutive convolution operations.
    """
    
    def __init__(self, in_ch, out_ch, kernel=3, pad=0, stride=1):
        super().__init__()
        self.operations = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel, padding=pad, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.operations(x)


class EncoderNetwork(nn.Module):
    """
    Lightweight VGG-style encoder for feature extraction from single modality.
    Uses progressively deeper feature maps with spatial downsampling.
    """
    
    def __init__(self, input_channels=1, latent_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Progressive encoding layers
        self.encode_stage1 = ConvolutionalBlock(input_channels, 64, kernel=3, pad=1)
        self.encode_stage2 = ConvolutionalBlock(64, 128, kernel=3, pad=1)
        self.encode_stage3 = ConvolutionalBlock(128, 256, kernel=3, pad=1)
        self.encode_stage4 = ConvolutionalBlock(256, latent_dim, kernel=3, pad=1)
    
    def forward(self, input_tensor):
        """
        Standard forward pass through encoding stages.
        
        Args:
            input_tensor: Input tensor with shape (batch, input_channels, H, W)
            
        Returns:
            torch.Tensor: Encoded features with shape (batch, latent_dim, H/8, W/8)
                (spatial dimensions reduced by 8x due to 3 max pooling operations)
        """
        # Stage 1: Initial feature extraction (no downsampling)
        stage1_out = self.encode_stage1(input_tensor)  # (batch, 64, H, W)
        
        # Stage 2: Downsample and encode
        stage2_out = F.max_pool2d(stage1_out, kernel_size=2)  # (batch, 64, H/2, W/2)
        stage2_out = self.encode_stage2(stage2_out)  # (batch, 128, H/2, W/2)
        
        # Stage 3: Further downsampling and encoding
        stage3_out = F.max_pool2d(stage2_out, kernel_size=2)  # (batch, 128, H/4, W/4)
        stage3_out = self.encode_stage3(stage3_out)  # (batch, 256, H/4, W/4)
        
        # Stage 4: Final downsampling and encoding to latent space
        stage4_out = F.max_pool2d(stage3_out, kernel_size=2)  # (batch, 256, H/8, W/8)
        final_features = self.encode_stage4(stage4_out)  # (batch, latent_dim, H/8, W/8)
        
        return final_features
    
    def forward_with_intermediates(self, input_tensor):
        """Forward pass that returns all intermediate feature maps"""
        stage1_out = self.encode_stage1(input_tensor)
        stage2_out = F.max_pool2d(stage1_out, kernel_size=2)
        stage2_out = self.encode_stage2(stage2_out)
        stage3_out = F.max_pool2d(stage2_out, kernel_size=2)
        stage3_out = self.encode_stage3(stage3_out)
        stage4_out = F.max_pool2d(stage3_out, kernel_size=2)
        final_features = self.encode_stage4(stage4_out)
        return final_features, stage3_out, stage2_out, stage1_out


class DualEncoderClassifier(nn.Module):
    """
    Classification model using two independent encoders with late fusion.
    Suitable for lightweight multi-modal satellite image classification.
    """
    
    def __init__(self, encoder1, encoder2, latent_dim=128, num_categories=10):
        super().__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classification_layer = nn.Linear(latent_dim * 2, num_categories)
    
    def forward(self, input1, input2):
        """
        Process both inputs and produce classification output.
        
        Args:
            input1: First modality input tensor (batch, channels1, H, W)
            input2: Second modality input tensor (batch, channels2, H, W)
            
        Returns:
            torch.Tensor: Classification logits with shape (batch, num_categories)
        """
        # Extract features from each encoder independently
        features1 = self.encoder1(input1)  # (batch, latent_dim, H', W')
        features2 = self.encoder2(input2)  # (batch, latent_dim, H', W')
        
        # Concatenate features along channel dimension for fusion
        fused = torch.cat((features1, features2), dim=1)  # (batch, 2*latent_dim, H', W')
        
        # Global average pooling to reduce spatial dimensions to 1x1
        pooled = self.global_pool(fused)  # (batch, 2*latent_dim, 1, 1)
        
        # Flatten to 1D vector for classification
        flattened = torch.flatten(pooled, 1)  # (batch, 2*latent_dim)
        
        # Final classification layer
        output_logits = self.classification_layer(flattened)  # (batch, num_categories)
        
        return output_logits


# Segmentation components adapted from U-Net architecture
class DoubleConvBlock(nn.Module):
    """Two consecutive convolutions with batch norm and activation"""
    
    def __init__(self, channels_in, channels_out, mid_ch=None):
        super().__init__()
        if mid_ch is None:
            mid_ch = channels_out
        
        self.block = nn.Sequential(
            nn.Conv2d(channels_in, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, channels_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.block(x)


class DownsampleBlock(nn.Module):
    """Spatial downsampling via max pooling followed by convolutions"""
    
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(ch_in, ch_out)
        )
    
    def forward(self, x):
        return self.downsample(x)


class UpsampleBlock(nn.Module):
    """Spatial upsampling with skip connections"""
    
    def __init__(self, ch_in, ch_out, use_bilinear=True):
        super().__init__()
        
        if use_bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.convs = DoubleConvBlock(ch_in, ch_out, ch_in // 2)
        else:
            self.upsample = nn.ConvTranspose2d(ch_in, ch_in // 2, kernel_size=2, stride=2)
            self.convs = DoubleConvBlock(ch_in, ch_out)
    
    def forward(self, decoder_input, encoder_skip):
        """
        Upsample and concatenate with skip connection.
        
        Args:
            decoder_input: Features from decoder pathway (batch, ch_in, H, W)
            encoder_skip: Skip connection from encoder (batch, ch_skip, H', W')
            
        Returns:
            torch.Tensor: Upsampled and fused features (batch, ch_out, H', W')
        """
        # Upsample decoder features to match encoder skip connection size
        upsampled = self.upsample(decoder_input)
        
        # Handle potential size mismatches due to rounding in pooling/upsampling
        # Calculate padding needed to match encoder skip connection dimensions
        delta_h = encoder_skip.size()[2] - upsampled.size()[2]
        delta_w = encoder_skip.size()[3] - upsampled.size()[3]
        
        # Apply symmetric padding: half on each side, remainder on right/bottom
        upsampled = F.pad(
            upsampled,
            [delta_w // 2, delta_w - delta_w // 2,  # Left, Right
             delta_h // 2, delta_h - delta_h // 2]   # Top, Bottom
        )
        
        # Concatenate skip connection with upsampled features
        concatenated = torch.cat([encoder_skip, upsampled], dim=1)
        
        # Apply convolutions to fuse concatenated features
        return self.convs(concatenated)


class OutputConv(nn.Module):
    """Final 1x1 convolution for producing class predictions"""
    
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class SegmentationUNet(nn.Module):
    """
    Standard U-Net architecture for semantic segmentation.
    Features encoder-decoder structure with skip connections.
    """
    
    def __init__(self, input_channels, output_classes, use_bilinear=True):
        super().__init__()
        self.input_channels = input_channels
        self.output_classes = output_classes
        self.use_bilinear = use_bilinear
        
        # Encoder pathway
        self.input_conv = DoubleConvBlock(input_channels, 64)
        self.down_block1 = DownsampleBlock(64, 128)
        self.down_block2 = DownsampleBlock(128, 256)
        self.down_block3 = DownsampleBlock(256, 512)
        
        reduction_factor = 2 if use_bilinear else 1
        self.down_block4 = DownsampleBlock(512, 1024 // reduction_factor)
        
        # Decoder pathway
        self.up_block1 = UpsampleBlock(1024, 512 // reduction_factor, use_bilinear)
        self.up_block2 = UpsampleBlock(512, 256 // reduction_factor, use_bilinear)
        self.up_block3 = UpsampleBlock(256, 128 // reduction_factor, use_bilinear)
        self.up_block4 = UpsampleBlock(128, 64, use_bilinear)
        
        self.output_layer = OutputConv(64, output_classes)
    
    def forward(self, x):
        """Encode-decode forward pass with skip connections"""
        # Encoder
        enc1 = self.input_conv(x)
        enc2 = self.down_block1(enc1)
        enc3 = self.down_block2(enc2)
        enc4 = self.down_block3(enc3)
        bottleneck = self.down_block4(enc4)
        
        # Decoder
        dec = self.up_block1(bottleneck, enc4)
        dec = self.up_block2(dec, enc3)
        dec = self.up_block3(dec, enc2)
        dec = self.up_block4(dec, enc1)
        
        logits = self.output_layer(dec)
        return logits

