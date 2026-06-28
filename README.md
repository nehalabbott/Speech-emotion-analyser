
# Speech Emotion Analyzer

A machine learning application that predicts human emotions from **3-second speech recordings**. The model is trained using a combination of the **RAVDESS**, **CREMA-D**, and **TESS** speech emotion datasets, achieving robust performance across multiple speakers and recording conditions.

---

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

The model is trained using the following publicly available datasets:

- **RAVDESS**
- **CREMA-D**
- **TESS**

These datasets provide diverse emotional speech samples from multiple speakers, improving the model's generalization.

---

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
