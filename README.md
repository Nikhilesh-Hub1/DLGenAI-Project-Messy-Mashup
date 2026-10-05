# Messy Mashup - Music Genre Classification

## 📌 Project Overview

This project focuses on **music genre classification under noisy and mixed audio conditions**.

The project is based on the **Messy Mashup** Kaggle competition, where the goal is to classify music into one of **10 genres** using instrument-separated audio stems such as drums, bass, vocals, and other instruments.

The primary evaluation metric is **Macro F1**, making balanced performance across all genres important.

---

## 🎯 Objectives

- Build a robust music genre classification pipeline.
- Handle noisy and mixed audio conditions.
- Experiment with multiple deep learning architectures.
- Fine-tune a pretrained Audio Spectrogram Transformer (AST).
- Build CNN and CRNN models from scratch.
- Apply audio-specific data augmentation.
- Use multi-seed ensemble inference to improve prediction robustness.
- Generate a Kaggle-compatible submission file.

---

## 📂 Dataset

The dataset contains music organized by genre with separated audio stems:

- `drums.wav`
- `bass.wav`
- `vocals.wav`
- `others.wav`

The project also uses **ESC-50 environmental sounds** as an additional noise source for augmentation.

### Dataset Structure

```text
messy_mashup/
├── genres_stems/
│   ├── genre_1/
│   │   ├── song_XXXX/
│   │   │   ├── drums.wav
│   │   │   ├── bass.wav
│   │   │   ├── vocals.wav
│   │   │   └── others.wav
│   ├── genre_2/
│   └── ...
├── mashups/
├── test.csv
└── ESC-50-master/
    └── audio/
```

---

# 🧠 Models

## 1. Audio Spectrogram Transformer (AST)

The main model uses the pretrained:

```text
MIT/ast-finetuned-audioset-10-10-0.4593
```

The pretrained AST model is fine-tuned for the **10-class music genre classification task**.

### AST Configuration

| Parameter | Value |
|---|---:|
| Epochs | 4 |
| Batch Size | 4 |
| Learning Rate | 2e-5 |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Audio Sampling Rate | 16 kHz |
| Audio Duration | 12 seconds |
| Number of Seeds | 3 |
| Seeds | 42, 123, 999 |
| Inference Segments | 17 |

---

## 2. CNN

A CNN model is implemented **from scratch** using PyTorch.

The model uses:

- Mel Spectrogram features
- Convolutional layers
- ReLU activations
- Max pooling
- Adaptive average pooling
- Fully connected layers

---

## 3. CRNN

A CRNN model is also implemented from scratch.

It combines:

- CNN layers for spatial/audio feature extraction
- LSTM for sequential modelling
- Fully connected classification layer

The CRNN also uses Mel Spectrogram representations.

---

# 🔊 Data Augmentation

The AST training pipeline uses several augmentation strategies designed for the noisy nature of the competition.

### Cross-Song Stem Mixing

For each training sample, stems can be randomly selected from different songs belonging to the same genre.

This creates new combinations of:

```text
Drums + Bass + Vocals + Others
```

while preserving the genre label.

### ESC-50 Noise Augmentation

Environmental sounds from ESC-50 are randomly added to training samples with controlled amplitude.

```python
audio = audio + 0.08 * noise
```

### Random Cropping

For 12-second training segments, random portions of longer audio files are selected.

### Audio Normalization

Audio is normalized before being passed to the feature extractor.

---

# 🔬 AST Training Strategy

The AST model is trained using the **complete available training dataset**.

Three independent training runs are performed using different random seeds:

```text
Seed 42
Seed 123
Seed 999
```

Each trained model is saved separately:

```text
model_seed42.pth
model_seed123.pth
model_seed999.pth
```

This allows the final prediction to benefit from model diversity.

---

# 🤝 Ensemble Inference

The final AST prediction uses an ensemble of the three trained models.

For each test audio file:

1. Multiple 12-second segments are sampled.
2. Very low-energy segments are filtered using RMS.
3. Each segment is passed through all three AST models.
4. Softmax probabilities are calculated.
5. Predictions from the three models are averaged.
6. Predictions across valid segments are averaged.
7. The class with the highest final probability is selected.

### Ensemble Formula

```text
Final Probability =
    (Seed42 Probability
   + Seed123 Probability
   + Seed999 Probability) / 3
```

The final predicted class is:

```text
argmax(Final Probability)
```

---

# 📊 Experiment Tracking

The project uses **Weights & Biases (W&B)** for experiment tracking.

The training process records metrics such as:

- Training loss
- Validation accuracy for models using validation data
- Validation Macro F1
- Training phase
- Epoch information
- Inference time

This allows experiments and model performance to be tracked systematically.

---

# 📈 Evaluation

The primary competition metric is:

```text
Macro F1
```

Macro F1 gives equal importance to each genre and is therefore useful when evaluating multi-class genre classification performance.

For the CNN and CRNN experiments, validation performance is measured using:

- Accuracy
- Macro F1

The final AST system is evaluated through the Kaggle competition submission.

---

# 📤 Submission

The notebook generates:

```text
submission.csv
```

with the required columns:

```text
id,genre
```

Example:

```csv
id,genre
0001,rock
0002,jazz
0003,pop
```

The generated file can be submitted directly to the Kaggle competition.

---

# 🛠️ Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Librosa
- TorchAudio
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Weights & Biases
- Kaggle

---

# 📁 Project Workflow

```text
Dataset
   │
   ├── Genre-labelled stems
   │
   ├── Cross-song mixing
   │
   └── ESC-50 noise augmentation
           │
           ▼
     Audio preprocessing
           │
           ├───────────────┐
           ▼               ▼
         AST          Mel Spectrogram
           │               │
           │          ┌────┴────┐
           │          ▼         ▼
           │         CNN       CRNN
           │
           ▼
   3-Seed AST Training
   ├── Seed 
