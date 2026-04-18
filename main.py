import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


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

#feature extraction mfcc
def extract_features(audio, sr):
    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    delta_mfcc = librosa.feature.delta(mfcc).mean(axis=1)
    
    # Chroma
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)

    # Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    
    # Energy (RMS) - Helps with Anger vs Sadness
    rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)

    # Combine all features into one vector
    return np.hstack((mfcc_mean, mfcc_std, delta_mfcc, chroma, mel, rms))


#ravdess labeling
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

#load ravdess dataset
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

#crema labeling
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

#load crema dataset
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


#main body
from dotenv import load_dotenv
load_dotenv()

ravdess_path = os.getenv("RAVDESS_PATH")
crema_path = os.getenv("CREMA_PATH")
print("Loading datasets...")
X_ravdess, y_ravdess = load_ravdess(ravdess_path)
X_crema, y_crema = load_crema(crema_path)

print("RAVDESS:", len(X_ravdess))
print("CREMA:", len(X_crema))


# combine datasets
X = np.array(X_ravdess + X_crema)
y = y_ravdess + y_crema

print("Total:", len(X))

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

#scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#train using SVM with RBF kernel and class balancing
model = SVC(kernel='rbf', C=5, gamma='scale', class_weight='balanced')
model.fit(X_train, y_train)

#train accuracy
y_pred_train = model.predict(X_train)
print("Train Accuracy:", accuracy_score(y_train, y_pred_train))

#test accuracy
y_pred_test = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))


#test cross accuracy: train on mixed, test on crema
X_crema_scaled = scaler.transform(X_crema)
y_pred_crema = model.predict(X_crema_scaled)

print("Cross Accuracy (Train: Mixed → Test: CREMA):",
      accuracy_score(y_crema, y_pred_crema))


#Train on RAVDESS → Test on CREMA
sc_rav = StandardScaler()
X_rav_scaled = sc_rav.fit_transform(X_ravdess)
X_cre_scaled_for_rav = sc_rav.transform(X_crema) # Use RAVDESS scaler on CREMA

model_rav = SVC(kernel='rbf', C=5, gamma='scale', class_weight='balanced')
model_rav.fit(X_rav_scaled, y_ravdess)

y_pred_cre = model_rav.predict(X_cre_scaled_for_rav)
print("Cross Accuracy (RAV -> CRE):", accuracy_score(y_crema, y_pred_cre))


#Train on CREMA → Test on RAVDESS
sc_cre = StandardScaler()
X_cre_scaled = sc_cre.fit_transform(X_crema)
X_rav_scaled_for_cre = sc_cre.transform(X_ravdess) # Use CREMA scaler on RAVDESS

model_cre = SVC(kernel='rbf', C=5, gamma='scale', class_weight='balanced')
model_cre.fit(X_cre_scaled, y_crema)

y_pred_rav = model_cre.predict(X_rav_scaled_for_cre)
print("Cross Accuracy (CRE -> RAV):", accuracy_score(y_ravdess, y_pred_rav))