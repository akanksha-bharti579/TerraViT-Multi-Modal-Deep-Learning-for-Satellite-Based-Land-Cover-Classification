# TerraViT: Manual Run Guide

This guide provides step-by-step instructions to run the TerraViT project manually.

---

## Prerequisites

- **Python 3.8 or higher** (check with: `python --version`)
- **pip** (Python package installer)
- **Windows/Linux/macOS** (tested on Windows)

---

## Step 1: Navigate to Project Directory

Open your terminal/command prompt and navigate to the project:

**Windows (PowerShell):**
```powershell
cd path/to/TerraViT/TerraViT-main
```

**Windows (Command Prompt):**
```cmd
cd path/to/TerraViT/TerraViT-main
```

**Linux/macOS:**
```bash
cd /path/to/TerraViT/TerraViT-main
```

---

## Step 2: Verify Python Installation

Check if Python is installed and accessible:

```bash
python --version
```

**Expected output:** `Python 3.8.x` or higher

If `python` doesn't work, try `python3`:
```bash
python3 --version
```

---

## Step 3: Install Dependencies

Install all required packages from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

**Or if you have pip directly:**
```bash
pip install -r requirements.txt
```

**Note:** If `albumentations` installation fails (requires C++ build tools), the project will still work - the data module will be optional.

**Expected output:**
- Packages being downloaded and installed
- No errors (warnings are okay)

---

## Step 4: Verify Installation

Test if PyTorch is installed correctly:

```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

**Expected output:** `PyTorch version: 2.x.x`

---

## Step 5: Run Quick Start Demo

This is the simplest demo that tests the basic functionality:

```bash
python examples/quick_start_demo.py
```

**What it does:**
- Creates synthetic Sentinel-2 satellite data
- Initializes a TerraViT model
- Performs inference
- Saves visualization to `terravit_demo_output.png`

**Expected output:**
```
======================================================================
TerraViT Quick Start Demo - Model Testing
======================================================================
...
[OK] Model initialized successfully
...
Demo completed successfully!
```

**Output file:** `terravit_demo_output.png` (created in the project root)

---

## Step 6: Run Advanced Dual-Stream Demo

This demonstrates the full multi-modal fusion architecture:

```bash
python examples/advanced_swin_demo.py
```

**What it does:**
- Creates synthetic Sentinel-1 (SAR) and Sentinel-2 (optical) data
- Initializes dual-stream ResNet architecture
- Performs multi-modal fusion
- Shows ablation study (S1-only vs S2-only vs Fusion)
- Saves comprehensive visualization

**Expected output:**
```
================================================================================
                    TerraViT Advanced Demo
             Swin Transformer + Dual-Stream Fusion
================================================================================
...
✓ Dual-Stream ResNet initialized
...
✨ Advanced demo completed successfully! ✨
```

**Output file:** `terravit_advanced_demo.png` (created in the project root)

---

## Step 7: Test Model Import (Optional)

Verify that you can import and use the model in your own code:

**Create a test file:** `test_model.py`

```python
import torch
from src.models import BimodalResNetClassifier

# Initialize model
model = BimodalResNetClassifier(
    modality1_channels=2,   # Sentinel-1 (VV + VH)
    modality2_channels=13,  # Sentinel-2 (13 bands)
    output_classes=8
)

print("Model initialized successfully!")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create synthetic data
s1_image = torch.randn(1, 2, 224, 224)   # Sentinel-1
s2_image = torch.randn(1, 13, 224, 224) # Sentinel-2

# Run inference
model.eval()
with torch.no_grad():
    input_dict = {"s1": s1_image, "s2": s2_image}
    prediction = model(input_dict)
    predicted_class = torch.argmax(prediction, dim=1)

print(f"Prediction shape: {prediction.shape}")
print(f"Predicted class: {predicted_class.item()}")
```

**Run the test:**
```bash
python test_model.py
```

**Expected output:**
```
Model initialized successfully!
Total parameters: 47,077,064
Prediction shape: torch.Size([1, 8])
Predicted class: 3
```

---

## Troubleshooting

### Issue 1: "ModuleNotFoundError: No module named 'torch'"

**Solution:** Install PyTorch:
```bash
python -m pip install torch torchvision
```

### Issue 2: "ModuleNotFoundError: No module named 'albumentations'"

**Solution:** This is optional. The project works without it. If you want to install it:
- **Windows:** Requires Visual C++ Build Tools (download from Microsoft)
- **Linux/macOS:** Usually installs without issues

### Issue 3: "UnicodeEncodeError" on Windows

**Solution:** The encoding fix is already in the advanced demo. If you see this in other scripts, add at the top:
```python
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
```

### Issue 4: "python: command not found"

**Solution:** 
- Try `python3` instead of `python`
- Or add Python to your system PATH
- Or use full path to your Python installation

### Issue 5: Script runs but no output file created

**Solution:** 
- Check if matplotlib is installed: `python -m pip install matplotlib`
- Check the current directory - files are saved in the directory where you run the script

---

## Quick Reference Commands

```bash
# Navigate to project
cd path/to/TerraViT/TerraViT-main

# Install dependencies
python -m pip install -r requirements.txt

# Run quick demo
python examples/quick_start_demo.py

# Run advanced demo
python examples/advanced_swin_demo.py

# Check Python version
python --version

# Check PyTorch installation
python -c "import torch; print(torch.__version__)"
```

---

## Using the Model in Your Own Code

### Basic Usage Example

```python
import torch
from src.models import BimodalResNetClassifier

# 1. Initialize model
model = BimodalResNetClassifier(
    modality1_channels=2,   # Sentinel-1 channels
    modality2_channels=13,  # Sentinel-2 channels
    output_classes=8        # 8 land cover classes
)

# 2. Prepare your data
# Your Sentinel-1 data (batch_size, 2, height, width)
s1_data = torch.randn(4, 2, 224, 224)

# Your Sentinel-2 data (batch_size, 13, height, width)
s2_data = torch.randn(4, 13, 224, 224)

# 3. Run inference
model.eval()  # Set to evaluation mode
with torch.no_grad():
    input_dict = {"s1": s1_data, "s2": s2_data}
    predictions = model(input_dict)  # Shape: (batch_size, 8)
    
    # Get predicted classes
    predicted_classes = torch.argmax(predictions, dim=1)
    
    # Get probabilities
    probabilities = torch.softmax(predictions, dim=1)

print(f"Predictions: {predicted_classes}")
print(f"Probabilities shape: {probabilities.shape}")
```

### Land Cover Classes

```python
LANDCOVER_CLASSES = {
    0: "Forest",
    1: "Shrubland",
    2: "Grassland",
    3: "Wetlands",
    4: "Croplands",
    5: "Urban/Built-up",
    6: "Barren",
    7: "Water"
}

# Use it:
predicted_class_idx = predicted_classes[0].item()
class_name = LANDCOVER_CLASSES[predicted_class_idx]
print(f"Predicted: {class_name}")
```

---

## Next Steps

1. **Run the demos** to see the project in action
2. **Explore the code** in `src/models/` to understand the architecture
3. **Modify the examples** to use your own data
4. **Read the README.md** for detailed documentation
5. **Check PRESENTATION_SLIDES.md** for project overview

---

## Need Help?

- Check the README.md for detailed documentation
- Review example scripts in `examples/` directory
- Check error messages - they often indicate what's missing
- Verify all dependencies are installed: `python -m pip list`

---

**Last Updated:** Current Date
**Project Status:** ✅ Working and tested

