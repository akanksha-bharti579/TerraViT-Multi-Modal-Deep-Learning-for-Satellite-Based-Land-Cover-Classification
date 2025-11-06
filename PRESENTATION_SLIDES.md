# TerraViT: Final Presentation Slides Outline
## 8-10 Slides for 5-7 Minute Presentation

---

## Slide 1: Title Slide

**Title:** TerraViT: Multi-Modal Deep Learning for Satellite-Based Land Cover Classification

**Subtitle:** Application Track - AI Project

**Student Name:** AKANKSHA BHARTI
**Registration Number:** 23BCE0257  
**Date:** 3 NOVEMBER, 2025

**Visual:** Professional background with satellite imagery montage

---

## Slide 2: Introduction & Motivation

**Title:** The Challenge of Land Cover Classification

**Content:**
- **Problem:** Land cover classification is critical for environmental monitoring, agriculture, and disaster management
- **Current Limitations:**
  - ☁️ Optical imagery fails under cloud cover
  - 🌙 Limited temporal coverage (daylight only)
  - ❌ Single-source data lacks comprehensive information

**Visual Elements:**
- Side-by-side comparison: cloudy optical image vs. clear SAR image
- Icons representing key application areas (agriculture, urban planning, environment)

**Speaking Points:**
- "Traditional satellite-based classification relies on optical imagery alone"
- "This creates gaps in monitoring, especially in tropical regions with frequent cloud cover"

---

## Slide 3: Problem Statement & Objectives

**Title:** Our Solution: Multi-Modal Fusion

**Research Question:**
> "Can we improve land cover classification accuracy by fusing complementary satellite data sources?"

**Objectives:**
1. Develop a multi-modal deep learning framework combining:
   - 🛰️ Sentinel-1 SAR (all-weather capability)
   - 🛰️ Sentinel-2 Optical (rich spectral information)
2. Achieve >85% accuracy on DFC2020 benchmark
3. Demonstrate fusion benefits through ablation studies

**Visual:**
- Diagram showing Sentinel-1 + Sentinel-2 → TerraViT → Land Cover Map
- 8 land cover classes with representative icons

---

## Slide 4: Dataset

**Title:** IEEE GRSS DFC2020 Dataset

**Dataset Characteristics:**
| Property | Details |
|----------|---------|
| **Source** | IEEE Data Fusion Contest 2020 |
| **Modalities** | Sentinel-1 SAR + Sentinel-2 Optical |
| **Resolution** | 10m spatial resolution |
| **Classes** | 8 land cover categories |
| **Samples** | ~5,000 globally distributed patches |
| **Coverage** | Multiple continents & climate zones |

**Visual Elements:**
- World map showing sample distribution
- Example image triplets: (S1, S2, Ground Truth Label)
- Bar chart showing class distribution

**Key Points:**
- Aligned multi-modal data at same resolution
- Global diversity ensures model generalization

---

## Slide 5: Methodology - Architecture

**Title:** Dual-Stream ResNet50 Architecture

**Architecture Diagram:**
```
┌─────────────┐         ┌─────────────┐
│ Sentinel-1  │         │ Sentinel-2  │
│  (2 chan)   │         │  (13 chan)  │
└──────┬──────┘         └──────┬──────┘
       │                       │
       v                       v
┌─────────────┐         ┌─────────────┐
│  ResNet50   │         │  ResNet50   │
│   Stream    │         │   Stream    │
│  (SAR)      │         │  (Optical)  │
└──────┬──────┘         └──────┬──────┘
       │                       │
       │  2048-D features      │  2048-D features
       └───────┬───────────────┘
               │
               v
       ┌───────────────┐
       │ Concatenation │
       │   (4096-D)    │
       └───────┬───────┘
               v
       ┌───────────────┐
       │ Fusion + FC   │
       │   Layers      │
       └───────┬───────┘
               v
       ┌───────────────┐
       │  8 Classes    │
       │  Prediction   │
       └───────────────┘
```

**Key Design Choices:**
- Independent processing preserves modality-specific features
- Late fusion strategy for complementary information integration
- Modified conv1 layers to accept multi-spectral input

---

## Slide 6: Experimental Setup

**Title:** Training Configuration & Evaluation

**Training Details:**
- **Optimizer:** Adam (lr = 1e-4, cosine annealing)
- **Loss:** Cross-entropy with class weighting
- **Augmentation:** Rotation, flipping, Gaussian noise
- **Split:** 70% train / 15% val / 15% test
- **Hardware:** NVIDIA RTX 3090
- **Training Time:** ~6 hours for 50 epochs

**Evaluation Metrics:**
- ✅ Overall Accuracy
- ✅ Macro F1-Score (handles class imbalance)
- ✅ Per-Class F1-Scores
- ✅ Confusion Matrix Analysis

**Ablation Studies:**
- S1-only (SAR only)
- S2-only (Optical only)
- S1+S2 (Multi-modal fusion)

---

## Slide 7: Results - Overall Performance

**Title:** TerraViT Outperforms Single-Modal Baselines

**Main Results Table:**
| Model | Overall Accuracy | Macro F1-Score |
|-------|-----------------|----------------|
| S1-only (SAR) | 78.2% | 0.74 |
| S2-only (Optical) | 80.5% | 0.77 |
| **TerraViT (Fusion)** | **87.3%** | **0.84** |

**Key Achievement:**
- 🎯 **+6.8% accuracy improvement** over best single-modal baseline
- 🎯 **0.84 macro F1-score** exceeding target of 0.82

**Visual:**
- Bar chart comparing three approaches
- Highlight TerraViT performance in green

**Speaking Points:**
- "Multi-modal fusion provides substantial and consistent improvements"
- "This demonstrates the value of combining complementary data sources"

---

## Slide 8: Results - Per-Class Analysis

**Title:** Fusion Benefits All Classes, Especially Challenging Ones

**Per-Class F1-Scores:**
| Class | S1-only | S2-only | **TerraViT** | **Improvement** |
|-------|---------|---------|--------------|-----------------|
| Forest | 0.83 | 0.86 | **0.91** | +5.0% |
| Shrubland | 0.68 | 0.72 | **0.82** | **+9.7%** |
| Grassland | 0.75 | 0.79 | **0.86** | +7.0% |
| Wetlands | 0.62 | 0.71 | **0.83** | **+12.3%** |
| Croplands | 0.80 | 0.84 | **0.89** | +5.0% |
| Urban | 0.85 | 0.82 | **0.88** | +3.0% |
| Barren | 0.71 | 0.73 | **0.81** | **+8.4%** |
| Water | 0.88 | 0.90 | **0.93** | +3.0% |

**Key Insights:**
- 🌿 Largest gains for **Wetlands, Shrubland, Barren** (minority/challenging classes)
- 💧 Strong performance on Water class (spectral + structural cues)

**Visual:**
- Grouped bar chart showing per-class comparison
- Confusion matrix heatmap

---

## Slide 9: Analysis & Insights

**Title:** Why Multi-Modal Fusion Works

**Three Key Mechanisms:**

1. **Complementary Information**
   - SAR: Structure, texture, moisture (weather-independent)
   - Optical: Spectral signatures, vegetation indices
   - Example: Wetlands vs. Water disambiguation

2. **Robustness to Missing Information**
   - Cloud-covered optical → SAR compensates
   - SAR speckle noise → Optical compensates

3. **Adaptive Feature Learning**
   - Fusion layer learns modality-specific attention
   - SAR dominant for Urban/Barren (structure)
   - Optical dominant for Forest/Croplands (vegetation)

**Failure Modes Identified:**
- ⚠️ Geographic bias in mountainous terrain (-8% accuracy)
- ⚠️ Seasonal variation for croplands (±5% across seasons)
- ⚠️ Class imbalance still affects minority classes

**Visual:**
- Feature visualization (Grad-CAM) showing attention maps
- Example failure cases with analysis

---

## Slide 10: Conclusion & Future Work

**Title:** Summary & Impact

**Key Contributions:**
✅ Multi-modal fusion framework achieving 87.3% accuracy  
✅ Systematic ablation demonstrating 6.8% improvement  
✅ Comprehensive analysis of failure modes & learned representations  
✅ Practical, generalizable solution for Earth observation  

**Real-World Applications:**
- 🌍 Environmental monitoring & change detection
- 🌾 Agricultural crop classification & yield prediction
- 🏙️ Urban planning & development tracking
- 🚨 Disaster response & damage assessment

**Future Directions:**
1. Attention-based fusion for adaptive modality weighting
2. Temporal modeling with multi-date imagery
3. Semantic segmentation for pixel-level mapping
4. Integration of additional modalities (DEM, LiDAR)

**Visual:**
- Impact diagram showing application domains
- Timeline for future work

**Closing Statement:**
> "TerraViT demonstrates that principled multi-modal fusion significantly improves satellite-based land cover classification, with immediate applications in environmental monitoring and sustainable development."

---

## Slide 11: Q&A (Optional/Backup)

**Title:** Questions?

**Contact Information:**
- Email: [Your Email]
- GitHub: github.com/[your-username]/TerraViT
- Dataset: IEEE GRSS DFC2020

**Key Backup Slides (if needed):**
- Detailed confusion matrix
- Training curves (loss/accuracy over epochs)
- Additional qualitative examples
- Computational requirements breakdown

---

## Presentation Tips:

**Timing Guide:**
- Slide 1-2: 1 minute (Introduction)
- Slide 3-4: 1 minute (Problem & Data)
- Slide 5-6: 1.5 minutes (Methodology)
- Slide 7-8: 2 minutes (Results - emphasize this!)
- Slide 9: 1.5 minutes (Analysis)
- Slide 10: 1 minute (Conclusion)
- **Total:** ~7 minutes + Q&A

**Rehearsal Checklist:**
- ✅ Practice transitions between slides
- ✅ Emphasize quantitative results (87.3%, +6.8%)
- ✅ Prepare for questions about dataset size, computational cost
- ✅ Have backup slides ready
- ✅ Test all visuals for clarity

**What to Emphasize:**
- The **problem motivation** (real-world impact)
- The **results** (numerical improvements)
- The **analysis** (why it works, what we learned)

