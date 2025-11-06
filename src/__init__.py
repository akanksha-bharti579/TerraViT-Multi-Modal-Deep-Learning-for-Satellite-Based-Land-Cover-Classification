"""
TerraViT: Multi-modal Deep Learning for Earth Observation

A comprehensive framework for satellite imagery analysis using
self-supervised learning and multi-modal fusion techniques.
"""

__version__ = "1.0.0"
__author__ = "Computer Vision Research Group"

from . import models
from . import training

# Data module requires albumentations - make it optional
try:
    from . import data
    __all__ = ["models", "data", "training"]
except ImportError:
    # albumentations not installed, skip data module
    __all__ = ["models", "training"]

