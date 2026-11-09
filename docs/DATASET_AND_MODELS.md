# LeafSense Models and Dataset Transparency

This document provides transparent technical details regarding the four deep neural network architectures, the 17 supported crop disease classes, ensemble weighting mathematics, and dataset lineage.

---

## 1. Supported Crops and Disease Classes (17 Classes)

LeafSense supports **17 specific plant disease and pathogen classes** across **5 economically vital agricultural crops**. The exact class mappings from `backend/utils.py` are detailed below:

| Crop | Class Label Identifier | Pathogen Type | Typical Field Symptoms |
|:---|:---|:---|:---|
| **Apple** | `Apple___Apple_scab` | Fungus (*Venturia inaequalis*) | Olive-green to black velvety spots on leaves; cracked fruit lesions |
| **Apple** | `Apple___Black_rot` | Fungus (*Botryosphaeria obtusa*) | "Frog-eye" circular leaf spots with purple margins; firm rotting fruit |
| **Apple** | `Apple___Cedar_apple_rust` | Fungus (*Gymnosporangium juniperi-virginianae*) | Bright orange-yellow circular spots with gelatinous horn projections |
| **Corn (Maize)** | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | Fungus (*Cercospora zeae-maydis*) | Rectangular tan-to-gray lesions bounded by leaf veins |
| **Corn (Maize)** | `Corn_(maize)___Common_rust_` | Fungus (*Puccinia sorghi*) | Oval cinnamon-brown powdery pustules scattered across both leaf surfaces |
| **Corn (Maize)** | `Corn_(maize)___Northern_Leaf_Blight` | Fungus (*Exserohilum turcicum*) | Long elliptical grayish-green cigar-shaped lesions on leaves |
| **Grape** | `Grape___Black_rot` | Fungus (*Guignardia bidwellii*) | Small circular reddish-brown leaf spots; shriveled black mummified berries |
| **Grape** | `Grape___Esca_(Black_Measles)` | Fungal complex | "Tiger-stripe" interveinal chlorosis and necrosis; spotted berries |
| **Grape** | `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` | Fungus (*Pseudocercospora cladosporioides*) | Irregular dark brown spots with halo; premature autumn defoliation |
| **Potato** | `Potato___Early_blight` | Fungus (*Alternaria solani*) | Concentric dark brown rings on older foliage forming "target board" marks |
| **Potato** | `Potato___Late_blight` | Oomycete (*Phytophthora infestans*) | Water-soaked pale lesions turning dark; white fungal down under humid conditions |
| **Tomato** | `Tomato___Early_blight` | Fungus (*Alternaria solani*) | Concentric target spots on bottom leaves; collar rot on seedling stems |
| **Tomato** | `Tomato___Late_blight` | Oomycete (*Phytophthora infestans*) | Rapidly expanding dark greasy lesions; white mildew during wet weather |
| **Tomato** | `Tomato___Septoria_leaf_spot` | Fungus (*Septoria lycopersici*) | Small circular spots with dark brown borders and tiny black fruiting bodies |
| **Tomato** | `Tomato___Target_Spot` | Fungus (*Corynespora cassiicola*) | Brown lesions with concentric rings; attacks both foliage and green fruit |
| **Tomato** | `Tomato___Tomato_mosaic_virus` | Viral pathogen (ToMV) | Mottled light and dark green leaves; leaf puckering and severe stunting |
| **Tomato** | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Viral pathogen (TYLCV) | Upward curling of leaf margins; yellowing and dramatic reduction in fruit set |

---

## 2. Deep Learning Ensemble Architecture

To prevent single-model overfitting and enhance multi-disease co-occurrence recognition, LeafSense aggregates predictions across four distinct model families:

```
                          [ Input Foliage Image (224x224x3) ]
                                          |
          +-------------------+-----------+-----------+-------------------+
          |                   |                       |                   |
    [ ResNet50 ]       [ DenseNet121 ]         [ GhostNetV2 ]     [ AgriFusionNet ]
          |                   |                       |                   |
     Sigmoid Head        Sigmoid Head            Sigmoid Head        Sigmoid Head
    (Weight: 0.22)      (Weight: 0.30)          (Weight: 0.28)      (Weight: 0.20)
          |                   |                       |                   |
          +-------------------+-----------+-----------+-------------------+
                                          |
                        [ Weighted Sigmoid Aggregation ]
                                          |
                        [ Multi-Disease Predictions P >= 0.50 ]
                                          |
                        [ DenseNet121 Grad-CAM Localization ]
```

### Model Specifications

1. **DenseNet121 (Primary Classifier & Grad-CAM Backbone, Weight: 0.30):**
   - **Feature extractor:** DenseNet-121 with feature reuse across dense blocks (`features.denseblock4.denselayer16.conv2`).
   - **Output Head:** Adaptive average pooling followed by linear layer (`Linear(1024, 17)`).
   - **Role:** Generates class activation maps for visual explainability.

2. **GhostNetV2 (Edge-Optimized Backbone, Weight: 0.28):**
   - **Feature extractor:** GhostNetV2-100 from the `timm` library.
   - **Architecture:** Employs linear ghost operations to generate redundant feature maps cheaply, optimized for low latency and high parameter efficiency.
   - **Output Head:** `Linear(1280, 17)`.

3. **ResNet50 (Residual Feature Extractor, Weight: 0.22):**
   - **Feature extractor:** Standard 50-layer deep residual network with bottleneck residual blocks.
   - **Output Head:** `Linear(2048, 17)`.

4. **AgriFusionNet (Custom Hybrid Architecture, Weight: 0.20):**
   - **Feature extractor:** EfficientNet-B4 backbone (`models.efficientnet_b4(weights=None)`).
   - **Classifier Head:** Dual-stage projection head with Batch Normalization and Dropout:
     - `Linear(1792, 512)` $\to$ `BatchNorm1d` $\to$ `ReLU` $\to$ `Dropout(0.3)`
     - `Linear(512, 256)` $\to$ `BatchNorm1d` $\to$ `ReLU` $\to$ `Dropout(0.3)`
     - `Linear(256, 17)`

---

## 3. Dataset Lineage & Transparency Disclosure

### Training Lineage
- **Historical Heritage:** The training pipelines (`train_resnet.py`, `train_densenet.py`, `train_ghostnet.py`, `train_agri.py`) derive from public agricultural vision datasets including **PlantVillage** and **PlantDoc**.
- **Multi-Disease Synthesis:** Standard plant disease benchmarks are predominantly single-leaf, single-label images. Multi-disease co-occurrence training utilized synthetic image compositing and multi-label label vectors.

### Metric Transparency
> **Important Note:**
> - The exact train/validation/test split ratios, raw image counts per class, and independent real-world field validation metrics (precision, recall, F1 scores across diverse field lighting conditions) **have not been independently benchmarked or formally verified**.
> - Benchmark performance claims should not be assumed without empirical testing by the repository maintainer.
> - Diagnostic outputs from this system are **preliminary assistive indications**, not certified agronomical tests.
