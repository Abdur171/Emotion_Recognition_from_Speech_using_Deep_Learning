# Speech Emotion Recognition (SER) using Deep Learning

[![Domain](https://img.shields.io/badge/Domain-Speech%20Audio%20%26%20Deep%20Learning-FF6F00.svg)](#project-overview)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%26%20TorchAudio-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformer](https://img.shields.io/badge/Pretrained-Wav2Vec%202.0-F9A825.svg)](https://huggingface.co/)
[![Dataset](https://img.shields.io/badge/Dataset-RAVDESS%20Benchmark-blue.svg)](https://zenodo.org/record/1188976)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning framework for **Speech Emotion Recognition (SER)** comparing custom spectral convolutional neural networks against self-supervised foundation models (**Wav2Vec 2.0**). Evaluated under rigorous **speaker-independent cross-validation** on the RAVDESS benchmark across 4 emotional categories: *Neutral*, *Happy*, *Sad*, and *Angry*.

---

## Model Architectures & Design

```
   Raw Audio Waveform (16 kHz)
        │
        ├───────────────────────────────────────────────────────┐
        ▼                                                       ▼
┌─────────────────────────────────┐           ┌─────────────────────────────────┐
│     Log-Mel Spectrogram         │           │        Raw Waveform Samples     │
│   (64 Filterbanks, Librosa)     │           │        (Normalized Tensors)     │
└───────────────┬─────────────────┘           └─────────────────┬───────────────┘
                ▼                                               ▼
┌─────────────────────────────────┐           ┌─────────────────────────────────┐
│       Custom 2D CNN             │           │        Wav2Vec 2.0 Base         │
│  - 3x Conv2D (BN + ReLU + Pool) │           │  - 7-Layer CNN Feature Extractor│
│  - 2x Dense Linear Layers       │           │  - 12-Layer Transformer (Frozen)│
│  - 1.54M Trainable Parameters   │           │  - 94.6M Total / 230K Trainable │
└───────────────┬─────────────────┘           └─────────────────┬───────────────┘
                ▼                                               ▼
┌─────────────────────────────────┐           ┌─────────────────────────────────┐
│    Logits: [N, H, S, A]         │           │    Logits: [N, H, S, A]         │
└─────────────────────────────────┘           └─────────────────────────────────┘
```

### 1. Custom 2D Mel-Spectrogram CNN
- **Acoustic Front-End**: Converts raw 16 kHz audio into 64-band log-mel filterbank spectrograms.
- **Backbone**: 3 sequential convolutional blocks with batch normalization, ReLU activations, and max-pooling, feeding into a 2-layer fully connected classification head.
- **Parameters**: 1,535,556 (100% trainable).

### 2. Pretrained Wav2Vec 2.0 Foundation Model
- **Base Checkpoint**: `wav2vec2-base` via `torchaudio`.
- **Transfer Learning Strategy**: Frozen acoustic encoder (retaining self-supervised phonemic representations trained on 960 hours of unlabeled speech) coupled to an active pooling and projection classification head.
- **Parameters**: 94,601,220 total parameters (230,276 trainable classifier weights).

---

## Experimental Setup & Benchmark

- **Dataset**: [RAVDESS](https://zenodo.org/record/1188976) (Ryerson Audio-Visual Database of Emotional Speech and Song).
- **Emotions Analyzed**: 4-class taxonomy — *Neutral* (96 samples), *Happy* (192 samples), *Sad* (192 samples), *Angry* (192 samples) — total 672 samples.
- **Split Protocol**: **Strict speaker-independent partition** across 24 professional actors to guarantee no identity leakage:
  - **Training Set**: 16 actors (448 utterances)
  - **Validation Set**: 4 actors (112 utterances)
  - **Test Set**: 4 actors (112 utterances)

---

## Empirical Benchmark & Comparative Results

| Model Architecture | Total Params | Trainable Params | Overall Accuracy | Macro F1-Score | Angry F1 | Neutral F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Custom 2D CNN** | 1.54 M | 1.54 M (100%) | $61.01\% \pm 3.60\%$ | $\mathbf{0.5900 \pm 0.048}$ | 0.74 | **0.42** |
| **Wav2Vec 2.0 Base** | 94.60 M | **230 K (0.24%)** | $\mathbf{63.39\%}$ | $0.5436$ | $\mathbf{0.79}$ | 0.11 |

### Key Experimental Insights
1. **Transfer Learning Efficiency**: The pretrained Wav2Vec 2.0 model achieves higher overall accuracy ($63.39\%$) while training **$85\%$ fewer parameters** than the scratch CNN.
2. **Acoustic Arousal Specialization**: Wav2Vec 2.0 strongly outperforms on high-arousal acoustics (**Angry F1 = 0.79**), capturing intense vocal dynamics and prosodic energy.
3. **Class Imbalance & Neutral Challenge**: Both models struggled most with the *Neutral* emotion due to a $1:2$ sample imbalance ($96$ vs. $192$) and neutral speech acoustic proximity to conversational pretraining baselines.
4. **Generalization Stability**: The CNN model delivered more uniform across-class F1 performance across three random seed initializations.

---

## Error Analysis & Recommended Mitigation

To resolve the *Neutral* class imbalance without acquiring additional recordings, modify the classification loss function in `train_pretrained_final.py` to incorporate cost-sensitive inverse class weights:

```python
import torch
import torch.nn as nn

# Assign 2x penalization weight to under-represented Neutral class [Neutral, Happy, Sad, Angry]
class_weights = torch.tensor([2.0, 1.0, 1.0, 1.0]).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

---

## Execution Guide

### 1. Installation
```bash
git clone https://github.com/Abdur171/Project-Emotion-Recognition-from-Speech-using-Deep-Learning.git
cd Project-Emotion-Recognition-from-Speech-using-Deep-Learning

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch torchaudio torchvision librosa soundfile matplotlib seaborn pandas scikit-learn
```

### 2. Dataset Preparation
Download the RAVDESS Audio dataset into `data/` and execute speaker-independent partitioning:
```bash
python data_preparation.py
```

### 3. Training the Custom CNN Baseline
```bash
python train_cnn.py
```

### 4. Training the Wav2Vec 2.0 Transfer Learning Pipeline
```bash
python train_pretrained_final.py
```

---

## Repository Structure

```
Project-Emotion-Recognition-from-Speech-using-Deep-Learning/
├── data_preparation.py        # Speaker-independent train/val/test splitter
├── train_cnn.py               # Spectrogram extraction & CNN training pipeline
├── train_pretrained_final.py  # Wav2Vec 2.0 fine-tuning and evaluation driver
├── models/                    # Model definitions (CNN & Wav2Vec classification heads)
├── final_report.md            # Complete empirical laboratory report & statistics
└── README.md                  # Project showcase documentation
```

---

## Academic Context

Conducted as an academic deep learning research project investigating self-supervised representations versus handcrafted spectral features in paralinguistic audio signal processing.
