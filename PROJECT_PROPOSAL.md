# AI Project Proposal: TerraViT

## Project Title
**TerraViT: Multi-Modal Deep Learning for Satellite-Based Land Cover Classification**

---

## Team Members Details

**Student Name:** AKANKSHA BHARTI  
**Email ID:** akanshabharti12379@gmail.com
**Registration Number:** 23BCE0257  
**Contact Number:** 9234512463

**Team Size:** 1 student member

---

## Track
**Application Track**

---

## Problem Statement

Land cover classification is a critical task in environmental monitoring, agriculture, urban planning, and disaster management. Traditional methods rely on manual interpretation of satellite imagery, which is time-consuming, expensive, and not scalable for large geographic areas. Single-source satellite data (e.g., optical-only imagery) often fails under adverse weather conditions or lacks the structural information necessary for accurate classification. This creates a significant gap in our ability to monitor environmental changes in real-time, particularly in regions with frequent cloud cover or complex terrain.

This project addresses the challenge of robust, all-weather land cover classification by developing **TerraViT**, a multi-modal deep learning framework that fuses Sentinel-1 SAR (Synthetic Aperture Radar) and Sentinel-2 optical satellite data. By combining the all-weather capability of SAR with the rich spectral information of optical imagery, TerraViT enables accurate land cover classification across 8 categories (Forest, Shrubland, Grassland, Wetlands, Croplands, Urban/Built-up, Barren, and Water). This solution benefits environmental scientists, agricultural planners, disaster response teams, and policy makers who require accurate, timely land cover information for decision-making.

---

## Dataset

**Dataset:** IEEE GRSS DFC2020 (Data Fusion Contest 2020)

**Link:** https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest

**Description:** The DFC2020 dataset contains globally distributed samples with multi-modal satellite imagery:
- **Sentinel-1 SAR data:** Dual-polarization (VV and VH) SAR imagery at 10m resolution
- **Sentinel-2 Optical data:** 13 spectral bands covering visible, near-infrared, and shortwave infrared spectra at 10m resolution
- **Ground truth labels:** 8 land cover classes annotated for semantic segmentation
- **Coverage:** Multiple geographic regions across different continents and climate zones
- **Total samples:** Approximately 5,000+ labeled image patches

The dataset provides aligned Sentinel-1 and Sentinel-2 data, making it ideal for multi-modal fusion experiments.

---

## Methodology

**Model Architecture:**
1. **Dual-Stream ResNet50 Architecture:**
   - Stream 1: Processes Sentinel-1 SAR data (2 channels: VV + VH polarizations)
   - Stream 2: Processes Sentinel-2 optical data (13 spectral bands)
   - Modified convolutional layers to accept multi-spectral input
   - Late fusion strategy: Concatenate feature representations from both streams
   - Joint classification head for 8-class land cover prediction

2. **Advanced Architecture (Swin Transformer):**
   - Explore Swin Transformer backbone for improved feature extraction
   - Hierarchical attention mechanism for multi-scale feature learning

**Training Strategy:**
- Data preprocessing: Normalization, augmentation (rotation, flipping, color jittering)
- Cross-entropy loss function for multi-class classification
- Adam optimizer with learning rate scheduling
- Train/validation/test split: 70%/15%/15%
- Batch size: 32, Epochs: 50 with early stopping
- Evaluation metrics: Accuracy, F1-score (macro and per-class), Confusion Matrix

**Ablation Studies:**
- Compare single-modal (S1-only, S2-only) vs. multi-modal fusion performance
- Analyze contribution of each data modality to classification accuracy

---

## Evaluation Plan (Key Metrics)

**Primary Metric:**
- **Overall Accuracy:** Target ≥ 85% on the test set
- **Macro F1-Score:** Target ≥ 0.82 across all 8 land cover classes

**Secondary Metrics:**
- **Per-Class F1-Scores:** Evaluate performance on minority classes (e.g., Wetlands, Barren)
- **Confusion Matrix Analysis:** Identify specific class pairs that are commonly confused
- **Ablation Performance:** Demonstrate that multi-modal fusion outperforms single-modal approaches by at least 5% in accuracy

**Qualitative Evaluation:**
- Visual comparison of prediction maps vs. ground truth
- Analysis of failure cases (e.g., specific geographic regions or weather conditions)
- Feature visualization to understand learned representations

---

## Novelty/Contribution

**Application-Oriented Contributions:**

1. **Multi-Modal Fusion Implementation:** 
   - Comprehensive implementation of dual-stream architecture specifically designed for satellite imagery fusion
   - Systematic comparison of early, middle, and late fusion strategies

2. **Benchmark Evaluation:**
   - Thorough evaluation on the DFC2020 benchmark dataset
   - Direct comparison of three different model families:
     - Traditional CNN (ResNet50)
     - Vision Transformer (ViT)
     - Hierarchical Transformer (Swin Transformer)

3. **Ablation Studies:**
   - Detailed analysis demonstrating the value of multi-modal fusion over uni-modal approaches
   - Investigation of robustness under different weather conditions and geographic regions

4. **Practical Framework:**
   - End-to-end pipeline from raw satellite data to land cover predictions
   - Modular codebase enabling easy adaptation to other satellite image classification tasks
   - Comprehensive documentation and reproducible experiments

**Impact:**
This project demonstrates how multi-modal deep learning can significantly improve land cover classification accuracy, particularly in challenging conditions where single-source data fails. The framework is generalizable to other Earth observation tasks such as crop yield prediction, deforestation monitoring, and urban expansion tracking.

---

## Timeline

- **Week 1-2:** Data acquisition, preprocessing, and exploratory data analysis
- **Week 3-4:** Implementation of baseline single-modal models
- **Week 5-6:** Development of multi-modal fusion architecture
- **Week 7-8:** Training, hyperparameter tuning, and ablation experiments
- **Week 9:** Results analysis, visualization, and report writing
- **Week 10:** Final presentation preparation and submission

---

**Signature:** ___________________  
**Date:** ___________________

