import os
import librosa
import numpy as np

# -----------------------------
# PREPROCESS
# -----------------------------

def preprocess(file, duration=3):
    try:
        audio, sr = librosa.load(file, sr=16000, mono=True)

        # FIX LENGTH
        target_length = duration * sr

        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]

        audio = librosa.util.normalize(audio)

        return audio, sr
    except:
        return None, None
# -----------------------------
# FEATURE EXTRACTION (MFCC)
# -----------------------------
def extract_features(audio, sr):
    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=100)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    # Chroma
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)

    # Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)

    # Combine all features
    return np.hstack((mfcc_mean, mfcc_std, chroma, mel))

# -----------------------------
# RAVDESS LABEL
# -----------------------------
def get_label_ravdess(filename):
    code = filename.split("-")[2]

    mapping = {
        "01": "neutral",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fear",
        "07": "disgust",
        "08": "surprise"
    }

    return mapping.get(code)

# -----------------------------
# LOAD RAVDESS
# -----------------------------
def load_ravdess(path):
    X, y = [], []

    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".wav"):
                full_path = os.path.join(root, file)

                audio, sr = preprocess(full_path)

                if audio is None:
                    continue
                features = extract_features(audio, sr)
                label = get_label_ravdess(file)

                if label:
                    X.append(features)
                    y.append(label)

    return X, y

# -----------------------------
# CREMA LABEL
# -----------------------------
def get_label_crema(filename):
    emotion = filename.split("_")[2]

    mapping = {
        "ANG": "angry",
        "HAP": "happy",
        "SAD": "sad",
        "NEU": "neutral",
        "FEA": "fear",
        "DIS": "disgust"
    }

    return mapping.get(emotion)

# -----------------------------
# LOAD CREMA
# -----------------------------
def load_crema(path):
    X, y = [], []

    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".wav"):
                full_path = os.path.join(root, file)

                audio, sr = preprocess(full_path)

                if audio is None:
                    continue

                features = extract_features(audio, sr)
                label = get_label_crema(file)

                if label:
                    X.append(features)
                    y.append(label)

    return X, y


# =============================
# LOAD DATA
# =============================
from dotenv import load_dotenv
load_dotenv()

ravdess_path = os.getenv("RAVDESS_PATH")
crema_path = os.getenv("CREMA_PATH")
X_ravdess, y_ravdess = load_ravdess(ravdess_path)
X_crema, y_crema = load_crema(crema_path)

print("RAVDESS:", len(X_ravdess))
print("CREMA:", len(X_crema))

# Combine datasets
X = X_ravdess + X_crema
y = y_ravdess + y_crema

print("Total:", len(X))


# =============================
# TRAIN MODEL
# =============================
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y)

# scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train
#model = LogisticRegression(max_iter=3000, class_weight='balanced')
model = SVC(kernel='rbf', C=5, gamma='scale', class_weight='balanced')
model.fit(X_train, y_train)


# =============================
# TRAIN ACCURACY
# =============================
y_pred_train = model.predict(X_train)
print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
# =============================
# TEST ACCURACY
# =============================
y_pred_test = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))


# =============================
# CROSS-DATASET TEST
# =============================
X_crema_scaled = scaler.transform(X_crema)
y_pred_crema = model.predict(X_crema_scaled)

print("Cross Accuracy (Train: Mixed → Test: CREMA):",
      accuracy_score(y_crema, y_pred_crema))


# =============================
# BONUS: TRAIN ONLY RAVDESS → TEST CREMA
# =============================
scaler2 = StandardScaler()
X_ravdess_scaled = scaler2.fit_transform(X_ravdess)
X_crema_scaled2 = scaler2.transform(X_crema)

#model2 = LogisticRegression(max_iter=3000)
model2 = SVC(kernel='rbf', C=5, gamma='scale', class_weight='balanced')
model2.fit(X_ravdess_scaled, y_ravdess)

y_pred_cross = model2.predict(X_crema_scaled2)

print("Cross Accuracy (Train: RAVDESS → Test: CREMA):",
      accuracy_score(y_crema, y_pred_cross))



# =============================
# BONUS: TRAIN ONLY CREMA → TEST Ravdes
# =============================
scaler3 = StandardScaler()

X_crema_scaled = scaler3.fit_transform(X_crema)
X_ravdess_scaled3 = scaler3.transform(X_ravdess)

model3 = SVC(kernel='rbf', C=5, gamma='scale', class_weight='balanced')
model3.fit(X_crema_scaled, y_crema)

y_pred_cross = model3.predict(X_ravdess_scaled3)

print("Cross Accuracy (Train: CREMA → Test: RAVDESS):",
      accuracy_score(y_ravdess, y_pred_cross))