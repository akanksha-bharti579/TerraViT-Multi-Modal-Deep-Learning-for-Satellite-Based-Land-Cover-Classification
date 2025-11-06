#!/usr/bin/env python3
"""
TerraViT Quick Start Demo
Demonstrates loading and using TerraViT models without requiring datasets
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("TerraViT Quick Start Demo - Model Testing")
print("=" * 70)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
print(f"PyTorch Version: {torch.__version__}")

# Create synthetic Sentinel-2 data (13 bands, 224x224)
print("\n" + "=" * 70)
print("Creating Synthetic Sentinel-2 Data")
print("=" * 70)

# Configuration constants
batch_size = 2  # Number of samples to process
s2_channels = 13  # Sentinel-2 has 13 spectral bands (Coastal, Blue, Green, Red, NIR, SWIR, etc.)
image_size = 224  # Standard ImageNet input size (compatible with ResNet)

# Generate synthetic data (simulating satellite imagery)
# Using random normal distribution as placeholder for actual satellite data
# Shape: (batch_size, channels, height, width)
synthetic_s2 = torch.randn(batch_size, s2_channels, image_size, image_size).to(device)
print(f"[OK] Synthetic Sentinel-2 data shape: {synthetic_s2.shape}")
print(f"  - Batch size: {batch_size}")
print(f"  - Channels: {s2_channels} (spectral bands)")
print(f"  - Image size: {image_size}x{image_size} pixels")

# Create a simple ResNet-based classifier for land cover
print("\n" + "=" * 70)
print("Building TerraViT Classifier")
print("=" * 70)

class TerraViTSimpleClassifier(nn.Module):
    """
    Simplified TerraViT classifier for demonstration.
    Adapts ResNet50 for multi-spectral satellite imagery.
    
    Key modifications:
    - First conv layer accepts multi-spectral input (13 channels) instead of RGB (3)
    - Final fully-connected layer outputs 8 land cover classes
    """
    def __init__(self, in_channels=13, num_classes=8):
        super(TerraViTSimpleClassifier, self).__init__()
        
        # Load ResNet50 backbone (pretrained=False for demo, use True for better results)
        self.backbone = resnet50(pretrained=False)
        
        # Modify first conv layer to accept multi-spectral input
        # Original ResNet50 expects 3 RGB channels, we need 13 Sentinel-2 bands
        # Kernel size 7x7, stride 2x2 for initial downsampling, padding 3 for size preservation
        self.backbone.conv1 = nn.Conv2d(
            in_channels,  # 13 Sentinel-2 spectral bands
            64,  # Output channels (standard ResNet first layer)
            kernel_size=(7, 7),  # Standard ResNet conv1 kernel size
            stride=(2, 2),  # Downsample by 2x
            padding=(3, 3),  # Maintain spatial dimensions after stride
            bias=False  # BatchNorm follows, so bias not needed
        )
        
        # Modify final layer for land cover classification
        # ResNet50 outputs 2048-D features, map to num_classes
        self.backbone.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the classifier.
        
        Args:
            x: Input tensor of shape (batch, 13, 224, 224)
            
        Returns:
            Logits tensor of shape (batch, 8) for 8 land cover classes
        """
        return self.backbone(x)

# Initialize model
num_classes = 8  # DFC2020 dataset has 8 land cover classes
model = TerraViTSimpleClassifier(in_channels=s2_channels, num_classes=num_classes)
model = model.to(device)  # Move model to GPU if available
model.eval()  # Set to evaluation mode (disables dropout, uses batch norm stats)

print(f"[OK] Model initialized successfully")
print(f"  - Input channels: {s2_channels}")
print(f"  - Output classes: {num_classes}")
print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Land cover classes (from DFC2020 dataset)
class_names = {
    0: "Forest",
    1: "Shrubland", 
    2: "Grassland",
    3: "Wetlands",
    4: "Croplands",
    5: "Urban/Built-up",
    6: "Barren",
    7: "Water"
}

# Perform inference
print("\n" + "=" * 70)
print("Running Inference")
print("=" * 70)

# Disable gradient computation for inference (faster, less memory)
with torch.no_grad():
    # Forward pass: get raw logits from model
    outputs = model(synthetic_s2)  # Shape: (batch_size, num_classes)
    
    # Convert logits to probabilities using softmax
    # Softmax ensures probabilities sum to 1.0
    probabilities = torch.softmax(outputs, dim=1)  # Shape: (batch_size, num_classes)
    
    # Get predicted class (index with highest probability)
    predictions = torch.argmax(probabilities, dim=1)  # Shape: (batch_size,)

print(f"[OK] Inference completed")
print(f"  - Output shape: {outputs.shape}")
print(f"  - Predictions shape: {predictions.shape}")

# Display results
print("\n" + "=" * 70)
print("Prediction Results")
print("=" * 70)

for i in range(batch_size):
    pred_class = predictions[i].item()
    pred_prob = probabilities[i, pred_class].item()
    
    print(f"\nSample {i+1}:")
    print(f"  Predicted class: {pred_class} - {class_names[pred_class]}")
    print(f"  Confidence: {pred_prob:.2%}")
    print(f"  Top 3 predictions:")
    
    top3_probs, top3_indices = torch.topk(probabilities[i], 3)
    for j, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
        print(f"    {j+1}. {class_names[idx.item()]}: {prob.item():.2%}")

# Visualize synthetic data statistics
print("\n" + "=" * 70)
print("Data Statistics")
print("=" * 70)

print(f"Synthetic S2 data statistics:")
print(f"  - Mean: {synthetic_s2.mean().item():.4f}")
print(f"  - Std: {synthetic_s2.std().item():.4f}")
print(f"  - Min: {synthetic_s2.min().item():.4f}")
print(f"  - Max: {synthetic_s2.max().item():.4f}")

# Feature extraction example
print("\n" + "=" * 70)
print("Feature Extraction")
print("=" * 70)

# Extract features from intermediate layer using PyTorch hooks
# Hooks allow us to capture intermediate activations during forward pass
features = {}

def hook_fn(module, input, output):
    """
    Hook function to capture layer output.
    Called automatically during forward pass.
    """
    features['layer4'] = output  # Store layer4 output (final ResNet block)

# Register forward hook on layer4 (final ResNet block before global pooling)
# Hook will be called every time layer4 processes input
handle = model.backbone.layer4.register_forward_hook(hook_fn)

# Run forward pass to trigger hook
with torch.no_grad():
    _ = model(synthetic_s2)

# Remove hook to free memory (important for long-running scripts)
handle.remove()

if 'layer4' in features:
    feature_maps = features['layer4']
    print(f"[OK] Extracted features from backbone layer4:")
    print(f"  - Feature shape: {feature_maps.shape}")
    print(f"  - Feature dimensions: {feature_maps.shape[1]} channels")

# Summary
print("\n" + "=" * 70)
print("Demo Summary")
print("=" * 70)
print("""
Successfully demonstrated TerraViT capabilities:

1. Created synthetic multi-spectral satellite data (13 Sentinel-2 bands)
2. Built a TerraViT classifier with ResNet50 backbone
3. Adapted the model for multi-spectral input (13 channels)
4. Performed land cover classification inference
5. Extracted intermediate features for analysis

Next Steps:
   - Download real satellite datasets (DFC2020, SEN12MS)
   - Download pretrained models for better accuracy
   - Fine-tune on your specific use case
   - Explore the demo notebooks for more advanced features

Built on Research:
   Advanced self-supervised learning techniques for satellite imagery
   Multi-modal fusion strategies for earth observation
""")
print("=" * 70)

# Optional: Save a visualization
try:
    # Visualize the first few channels of the synthetic data
    # Create 2x3 grid for displaying 6 spectral bands
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('TerraViT: Synthetic Sentinel-2 Bands (First 6 bands)', 
                 fontsize=16, color='#2c3e50')
    
    # Display first 6 bands (out of 13 total)
    for i, ax in enumerate(axes.flat):
        if i < 6:
            # Extract channel i from first sample, move to CPU, convert to numpy
            channel_data = synthetic_s2[0, i].cpu().numpy()
            # Display with plasma colormap (good for visualizing satellite data)
            im = ax.imshow(channel_data, cmap='plasma')
            ax.set_title(f'Band {i+1}', color='#34495e')
            ax.axis('off')  # Remove axes for cleaner visualization
            # Add colorbar to show value scale
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Save figure with high DPI for quality
    plt.savefig('terravit_demo_output.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to: terravit_demo_output.png")
except Exception as e:
    # Gracefully handle visualization errors (e.g., if matplotlib backend unavailable)
    print(f"\nWarning: Could not save visualization: {e}")

print("\nDemo completed successfully!\n")

