
# Speech Emotion Analyzer

A machine learning application that predicts human emotions from **3-second speech recordings**. The model is trained using a combination of the **RAVDESS**, **CREMA-D**, and **TESS** speech emotion datasets, achieving robust performance across multiple speakers and recording conditions.

---
Demo Link: https://drive.google.com/file/d/1xoYOVdeBGChwVEcvEDRr3tYl3sT0suHS/view?usp=sharing
Confusion matrices: https://drive.google.com/drive/folders/1zutNj9CWar6mKSaYF5yBmOeUuMBjyMm9?usp=sharing

## Features

- 🎤 Emotion prediction from short audio clips
- 📊 Feature extraction using Librosa
- 🧠 LinearSVC classifier with balanced class weights
- 🌐 Interactive Streamlit web interface
- 📁 Supports WAV audio files
- ⚡ Fast inference with pre-trained models

---

## 🛠 Tech Stack

- Python
- Scikit-learn
- Librosa
- NumPy
- Pandas
- Streamlit
- Joblib

---

## 📂 Datasets

The model was trained and evaluated on a combined corpus of **10,887 speech samples** collected from three widely used emotion recognition datasets:

| Dataset | Samples |
|---------|---------:|
| RAVDESS | 1,056 |
| CREMA-D | 7,442 |
| TESS | 2,389 |
| **Total** | **10,887** |

These datasets contain recordings from multiple speakers expressing emotions such as happy, sad, angry, fearful, disgust, neutral, and surprise, providing a diverse benchmark for emotion recognition.

## 📊 Feature Engineering

The audio preprocessing pipeline extracts several acoustic features using **Librosa**:

- MFCCs (Mel Frequency Cepstral Coefficients)
- Chroma Features
- Mel Spectrogram
- Root Mean Square (RMS) Energy
- Spectral Contrast

Since audio clips vary in length, **Mean** and **Standard Deviation pooling** are applied to generate fixed-length feature vectors suitable for machine learning.

---

## 🤖 Model

The classifier used is:

- **Linear Support Vector Classifier (LinearSVC)**
- Balanced class weights to reduce bias caused by class imbalance
- StandardScaler for feature normalization

The trained models are saved as:

```
emotion_model.pkl
scaler.pkl
```

---

## 📁 Project Structure

```
Speech-Emotion-Analyzer/
│
├── app.py                 # Streamlit web application
├── main.py                # Training pipeline
├── requirements.txt       # Dependencies
├── emotion_model.pkl      # Trained classifier
├── scaler.pkl             # Feature scaler
├── .env.example           # Dataset path template
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <repository-url>
cd Speech-Emotion-Analyzer
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔧 Configuration

Create a `.env` file from the provided template:

```bash
cp .env.example .env
```

Update the dataset paths inside `.env`:

```
RAVDESS_PATH=...
CREMAD_PATH=...
TESS_PATH=...
```

---

## 🚀 Training the Model

Run:

```bash
python main.py
```
---

## 💻 Running the Application

Launch the Streamlit interface:

```bash
python -m streamlit run app.py
```

Upload a speech sample, and the application will predict the corresponding emotion.

---

## 📈 Workflow

```
Speech Audio
      │
      ▼
Feature Extraction
(MFCC • Chroma • Mel • RMS • Spectral Contrast)
      │
      ▼
Mean + Std Pooling
      │
      ▼
Feature Scaling
      │
      ▼
LinearSVC Model
      │
      ▼
Predicted Emotion
```
## 📈 Results

### Mixed-Dataset Evaluation

The model was trained using the combined dataset of **10,887 audio samples** and evaluated on a held-out test split.

| Metric | Score |
|--------|------:|
| Training Accuracy | **72.73%** |
| Training Macro F1 | **72.64%** |
| Test Accuracy | **57.97%** |
| Test Macro F1 | **58.00%** |

The balanced LinearSVC demonstrates reasonable generalization across multiple datasets despite differences in recording environments, speakers, and emotional expression styles.
## 🔬 Pooling Method Comparison

Different temporal pooling strategies were evaluated to convert variable-length audio features into fixed-length vectors.

| Pooling Method | Accuracy | Macro F1 |
|---------------|---------:|---------:|
| Mean | 40.92% | 41.24% |
| Max | 48.32% | 48.11% |
| Standard Deviation | 51.41% | 51.33% |
| **Mean + Standard Deviation** | **57.97%** | **58.00%** |

The **Mean + Standard Deviation** pooling strategy achieved the best overall performance and was selected for the final model.
## 🌍 Cross-Dataset Generalization

To evaluate robustness across different datasets, the model was trained on two datasets and tested on the third.

| Training Dataset(s) | Test Dataset | Accuracy | Macro F1 |
|--------------------|-------------|---------:|---------:|
| CREMA-D + TESS | RAVDESS | 29.45% | 22.21% |
| RAVDESS + TESS | CREMA-D | 21.41% | 11.90% |
| RAVDESS + CREMA-D | TESS | **59.77%** | **58.72%** |

These experiments demonstrate the domain shift that exists between different speech emotion datasets, highlighting the challenges of building models that generalize across recording conditions and speaker populations.
