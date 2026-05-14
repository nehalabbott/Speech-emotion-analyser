# Speech-emotion-analyser

To load dataset-
Copy .env.example to .env and update paths as per your local PC folder path of the dataset

We used mean and standard deviation pooling to convert variable-length audio features into fixed-size vectors, capturing both average behavior and variability over time.

run 
python main.py

for streamlit app
python -m streamlit run app.py



# Speech Emotion Recognition System 🎙️🧠

A robust **Speech Emotion Recognition (SER)** pipeline built using traditional machine learning and handcrafted acoustic features.

This project performs:

- Speaker-independent evaluation
- Cross-dataset generalization testing
- Pooling strategy comparison
- Emotion classification using SVM
- Feature caching for faster experimentation

The system is trained on multiple benchmark emotional speech datasets and focuses on **real-world generalization** rather than only achieving high train accuracy.

---

# 🚀 Features

- Multi-dataset training
- Speaker-independent splitting
- Cross-dataset evaluation
- Multiple pooling strategies
- MFCC + spectral handcrafted features
- Cached feature extraction
- Confusion matrix visualization
- Model + scaler export for deployment
- Fast experimentation workflow

---

# 📚 Datasets Used

## 1. RAVDESS
Ryerson Audio-Visual Database of Emotional Speech and Song

Supported emotions:
- Angry
- Happy
- Sad
- Neutral
- Fear
- Disgust

---

## 2. CREMA-D
Crowd-sourced Emotional Multimodal Actors Dataset

Supported emotions:
- Angry
- Happy
- Sad
- Neutral
- Fear
- Disgust

---

## 3. TESS
Toronto Emotional Speech Set

Supported emotions:
- Angry
- Happy
- Sad
- Neutral
- Fear
- Disgust

---

# 🧠 Emotion Classes

The model uses a common label space across all datasets:

```python
COMMON_EMOTIONS = [
    "angry",
    "happy",
    "sad",
    "neutral",
    "fear",
    "disgust"
]
```

Removed emotions:
- Calm
- Surprise

because all datasets do not contain them consistently.

---

# ⚙️ Audio Preprocessing

Each audio file undergoes:

## 1. Resampling

```python
sr = 16000
```

## 2. Mono conversion

Stereo → Mono

## 3. Fixed duration

Audio padded/truncated to:

```python
duration = 3 seconds
```

## 4. Normalization

Amplitude normalization using librosa.

---

# 🎵 Extracted Acoustic Features

The system extracts multiple handcrafted audio descriptors.

---

## MFCC (40 coefficients)

Captures vocal tract information.

```python
librosa.feature.mfcc()
```

---

## Delta MFCC

Captures temporal dynamics.

```python
librosa.feature.delta()
```

---

## Delta-Delta MFCC

Captures acceleration of speech patterns.

---

## Chroma Features

Captures harmonic pitch content.

```python
librosa.feature.chroma_stft()
```

---

## Log Mel Spectrogram

Frequency-energy representation.

```python
librosa.feature.melspectrogram()
```

---

## RMS Energy

Captures speech intensity.

```python
librosa.feature.rms()
```

---

## Zero Crossing Rate (ZCR)

Measures signal sign changes.

```python
librosa.feature.zero_crossing_rate()
```

---

## Spectral Contrast

Captures spectral peak-valley differences.

```python
librosa.feature.spectral_contrast()
```

---

# 🧮 Feature Pooling

Frame-level sequences are converted into fixed-length vectors using pooling.

Supported pooling strategies:

| Pooling | Description |
|---|---|
| Mean | Average feature value |
| Max | Maximum feature value |
| Std | Standard deviation |
| Mean + Std | Concatenation of mean and std |

Default:

```python
pooling = "mean+std"
```

---

# ⚡ Feature Caching

To avoid repeated heavy feature extraction:

```text
feature_cache/raw_features.joblib
```

stores:
- Raw feature sequences
- Labels
- Speaker groups

This drastically reduces runtime during experimentation.

---

# 🧪 Evaluation Strategies

# 1. Mixed Dataset Speaker-Independent Evaluation

All datasets combined together.

Train/test split uses:

```python
GroupShuffleSplit
```

where speakers are separated between train and test.

This prevents speaker leakage.

---

# 2. Same Dataset Evaluation

Performed independently on:
- RAVDESS
- CREMA-D
- TESS

Each dataset uses speaker-independent splitting.

---

# 3. Cross-Dataset Generalization

Leave-one-dataset-out experiments:

| Train | Test |
|---|---|
| CREMA + TESS | RAVDESS |
| RAVDESS + TESS | CREMA |
| RAVDESS + CREMA | TESS |

This tests domain generalization capability.

---

# 🤖 Model

Current classifier:

```python
LinearSVC
```

Configuration:

```python
LinearSVC(
    class_weight="balanced",
    random_state=42,
    max_iter=10000
)
```

---

# 📏 Feature Scaling

Standardization performed using:

```python
StandardScaler
```

---

# 📊 Evaluation Metrics

The project reports:

- Accuracy
- Macro F1-score
- Confusion Matrix

Macro F1 is used because:
- emotion classes may be imbalanced
- it evaluates all classes equally

---

# 📈 Confusion Matrix

Automatically generated using seaborn heatmaps.

Saved as:

```text
cm_*.png
```

Examples:
- `cm_mixed.png`
- `cm_rav_same.png`
- `cm_cre_same.png`

---

# 💾 Saved Models

The trained model is exported for deployment:

```text
emotion_model.pkl
scaler.pkl
```

Can be directly loaded into:
- Flask app
- FastAPI
- Streamlit
- Web deployment

---

# 📂 Project Structure

```text
project/
│
├── main.py
├── .env
├── emotion_model.pkl
├── scaler.pkl
│
├── feature_cache/
│   └── raw_features.joblib
│
├── confusion_matrices/
│
├── datasets/
│   ├── RAVDESS/
│   ├── CREMA-D/
│   └── TESS/
│
└── README.md
```

---

# 🔐 Environment Variables

Create a `.env` file:

```env
RAVDESS_PATH=path_to_ravdess
CREMA_PATH=path_to_crema
TESS_PATH=path_to_tess
```

---

# ▶️ Run The Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python main.py
```

---

# 📦 Dependencies

Main libraries used:

```text
librosa
numpy
scikit-learn
matplotlib
seaborn
python-dotenv
joblib
```

---

# 🔥 Current Challenges

The system currently faces:

- Overfitting on smaller datasets
- Dataset distribution mismatch
- Cross-corpus generalization difficulty
- High-dimensional handcrafted features

---

# 🚀 Planned Improvements

Future upgrades:

- RBF SVM
- PCA dimensionality reduction
- Data augmentation
- CNN/LSTM models
- Wav2Vec2 embeddings
- Attention-based architectures
- Hyperparameter tuning
- Ensemble learning

---

# 🏆 Key Learning Outcomes

This project demonstrates:

- Speech signal preprocessing
- Feature engineering
- Speaker-independent ML evaluation
- Cross-dataset robustness testing
- Emotion recognition pipelines
- Practical ML experimentation workflows

---

# 👨‍💻 Author

Built as an end-to-end Speech Emotion Recognition research & deployment project using Python and Scikit-learn.