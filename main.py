import os
import warnings
warnings.filterwarnings("ignore")
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# LOAD ENV VARIABLES
# =========================================================
load_dotenv()

RAVDESS_PATH = os.getenv("RAVDESS_PATH")
CREMA_PATH = os.getenv("CREMA_PATH")
TESS_PATH = os.getenv("TESS_PATH")

# =========================================================
# COMMON LABEL SET
# =========================================================
# We remove:
# - calm
# - surprise
# because CREMA-D doesn't contain them

COMMON_EMOTIONS = [
    "angry",
    "happy",
    "sad",
    "neutral",
    "fear",
    "disgust"
]

# =========================================================
# PREPROCESSING
# =========================================================
def preprocess(file_path, duration=3):
    try:
        audio, sr = librosa.load(
            file_path,
            sr=16000,
            mono=True
        )

        # FIX LENGTH
        target_length = duration * sr

        if len(audio) < target_length:
            audio = np.pad(
                audio,
                (0, target_length - len(audio))
            )
        else:
            audio = audio[:target_length]

        # NORMALIZATION
        audio = librosa.util.normalize(audio)

        return audio, sr

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None


# =========================================================
# RAW FEATURE SEQUENCE EXTRACTION
# =========================================================
def extract_raw_sequences(audio, sr):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    delta_mfcc = librosa.feature.delta(mfcc)

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr
    )

    rms = librosa.feature.rms(y=audio)

    zcr = librosa.feature.zero_crossing_rate(audio)

    spectral_contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=sr
    )

    # STACK ALL FEATURES
    return np.vstack((
        mfcc,
        delta_mfcc,
        chroma,
        mel,
        rms,
        zcr,
        spectral_contrast
    ))


# =========================================================
# POOLING METHODS
# =========================================================
def apply_pooling(sequence, pool_type):

    if pool_type == "mean":
        return np.mean(sequence, axis=1)

    elif pool_type == "max":
        return np.max(sequence, axis=1)

    elif pool_type == "std":
        return np.std(sequence, axis=1)

    elif pool_type == "mean+std":
        return np.hstack((
            np.mean(sequence, axis=1),
            np.std(sequence, axis=1)
        ))

    else:
        raise ValueError("Invalid pooling method")


# =========================================================
# RAVDESS LABELING
# =========================================================
def get_label_ravdess(filename):

    code = filename.split("-")[2]

    mapping = {
        "01": "neutral",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fear",
        "07": "disgust"
    }

    return mapping.get(code)


def get_speaker_ravdess(filename):
    return filename.split("-")[-1].split(".")[0]


# =========================================================
# CREMA-D LABELING
# =========================================================
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


def get_speaker_crema(filename):
    return filename.split("_")[0]


# =========================================================
# TESS LABELING
# =========================================================
def get_label_tess(filename):

    emotion = filename.split("_")[-1].split(".")[0]

    mapping = {
        "angry": "angry",
        "happy": "happy",
        "sad": "sad",
        "neutral": "neutral",
        "fear": "fear",
        "disgust": "disgust",
        "ps": None   # pleasant surprise removed
    }

    return mapping.get(emotion)


def get_speaker_tess(filename):

    # OAF or YAF
    return filename.split("_")[0]

# =========================================================
# LOAD RAVDESS
# =========================================================
def load_ravdess(path, pooling="mean+std"):

    X, y, groups = [], [], []

    for root, _, files in os.walk(path):

        for file in files:

            if file.endswith(".wav"):

                label = get_label_ravdess(file)

                if label not in COMMON_EMOTIONS:
                    continue

                full_path = os.path.join(root, file)

                audio, sr = preprocess(full_path)

                if audio is None:
                    continue

                raw_seq = extract_raw_sequences(audio, sr)

                pooled = apply_pooling(raw_seq, pooling)

                speaker = get_speaker_ravdess(file)

                X.append(pooled)
                y.append(label)
                groups.append(speaker)

    return np.array(X), np.array(y), np.array(groups)

# =========================================================
# LOAD TESS
# =========================================================
def load_tess(path, pooling="mean+std"):

    X, y, groups = [], [], []

    for root, _, files in os.walk(path):

        for file in files:

            if file.endswith(".wav"):

                label = get_label_tess(file)

                if label not in COMMON_EMOTIONS:
                    continue

                full_path = os.path.join(root, file)

                audio, sr = preprocess(full_path)

                if audio is None:
                    continue

                raw_seq = extract_raw_sequences(audio, sr)

                pooled = apply_pooling(raw_seq, pooling)

                speaker = get_speaker_tess(file)

                X.append(pooled)
                y.append(label)
                groups.append(speaker)

    return np.array(X), np.array(y), np.array(groups)

# =========================================================
# LOAD CREMA-D
# =========================================================
def load_crema(path, pooling="mean+std"):

    X, y, groups = [], [], []

    for root, _, files in os.walk(path):

        for file in files:

            if file.endswith(".wav"):

                label = get_label_crema(file)

                if label not in COMMON_EMOTIONS:
                    continue

                full_path = os.path.join(root, file)

                audio, sr = preprocess(full_path)

                if audio is None:
                    continue

                raw_seq = extract_raw_sequences(audio, sr)

                pooled = apply_pooling(raw_seq, pooling)

                speaker = get_speaker_crema(file)

                X.append(pooled)
                y.append(label)
                groups.append(speaker)

    return np.array(X), np.array(y), np.array(groups)


# =========================================================
# CONFUSION MATRIX
# =========================================================
def plot_confusion_matrix(
    y_true,
    y_pred,
    title,
    filename
):

    labels = sorted(list(set(y_true)))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.title(title)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.savefig(
        filename,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved confusion matrix: {filename}")


# =========================================================
# EVALUATION FUNCTION
# =========================================================
def evaluate_model(
    model,
    X_test,
    y_test,
    experiment_name,
    cm_filename
):

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

   
    print(experiment_name)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print("\n")

  #  print("\nClassification Report:\n")

   # print(classification_report(y_test, y_pred))

    plot_confusion_matrix(
        y_test,
        y_pred,
        experiment_name,
        cm_filename
    )

    return accuracy, macro_f1


# =========================================================
# LOAD DATASETS
# =========================================================
print("\nLoading datasets...\n")

X_rav, y_rav, g_rav = load_ravdess(
    RAVDESS_PATH,
    pooling="mean+std"
)

X_cre, y_cre, g_cre = load_crema(
    CREMA_PATH,
    pooling="mean+std"
)
X_tess, y_tess, g_tess = load_tess(
    TESS_PATH,
    pooling="mean+std"
)

print(f"TESS Samples : {len(X_tess)}")
print(f"RAVDESS Samples : {len(X_rav)}")
print(f"CREMA-D Samples : {len(X_cre)}")

# =========================================================
# COMBINE DATASETS
# =========================================================
X_all = np.vstack((
    X_rav,
    X_cre,
    X_tess
))

y_all = np.hstack((
    y_rav,
    y_cre,
    y_tess
))

groups_all = np.hstack((
    ["rav_" + g for g in g_rav],
    ["cre_" + g for g in g_cre],
    ["tess_" + g for g in g_tess]
))

print(f"Total Samples : {len(X_all)}")
print("\n")

# =========================================================
# TASK 1:
# SPEAKER-INDEPENDENT SPLIT
# =========================================================

gss = GroupShuffleSplit(
    test_size=0.2,
    n_splits=1,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(X_all, y_all, groups_all)
)

X_train = X_all[train_idx]
X_test = X_all[test_idx]

y_train = y_all[train_idx]
y_test = y_all[test_idx]

# =========================================================
# SCALING
# =========================================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================================================
# TRAIN LINEAR SVM
# =========================================================
model = LinearSVC(
    class_weight="balanced",
    random_state=42,
    max_iter=10000
)

model.fit(X_train, y_train)

# =========================================================
# EVALUATE
# =========================================================
evaluate_model(
    model,
    X_test,
    y_test,
    "Mixed Dataset Speaker-Independent Split",
    "cm_mixed.png"
)

# =========================================================
# TASK 2:
# LEAVE-ONE-DATASET-OUT CROSS DATASET EVALUATION
# =========================================================

print("\nStarting Task 2: Cross Dataset Generalization\n")

# =========================================================
# EXPERIMENT 1
# TRAIN: CREMA + TESS
# TEST : RAVDESS
# =========================================================

print("\n" + "="*60)
print("Train: CREMA + TESS -> Test: RAVDESS")
print("="*60)

X_train = np.vstack((X_cre, X_tess))
y_train = np.hstack((y_cre, y_tess))

X_test = X_rav
y_test = y_rav

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearSVC(
    class_weight="balanced",
    random_state=42,
    max_iter=10000
)

model.fit(X_train_scaled, y_train)

evaluate_model(
    model,
    X_test_scaled,
    y_test,
    "Train: CREMA+TESS -> Test: RAVDESS",
    "cm_cre_tess_to_rav.png"
)


# =========================================================
# EXPERIMENT 2
# TRAIN: RAVDESS + TESS
# TEST : CREMA
# =========================================================

print("\n" + "="*60)
print("Train: RAVDESS + TESS -> Test: CREMA")
print("="*60)

X_train = np.vstack((X_rav, X_tess))
y_train = np.hstack((y_rav, y_tess))

X_test = X_cre
y_test = y_cre

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearSVC(
    class_weight="balanced",
    random_state=42,
    max_iter=10000
)

model.fit(X_train_scaled, y_train)

evaluate_model(
    model,
    X_test_scaled,
    y_test,
    "Train: RAVDESS+TESS -> Test: CREMA",
    "cm_rav_tess_to_cre.png"
)


# =========================================================
# EXPERIMENT 3
# TRAIN: RAVDESS + CREMA
# TEST : TESS
# =========================================================

print("\n" + "="*60)
print("Train: RAVDESS + CREMA -> Test: TESS")
print("="*60)

X_train = np.vstack((X_rav, X_cre))
y_train = np.hstack((y_rav, y_cre))

X_test = X_tess
y_test = y_tess

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearSVC(
    class_weight="balanced",
    random_state=42,
    max_iter=10000
)

model.fit(X_train_scaled, y_train)

evaluate_model(
    model,
    X_test_scaled,
    y_test,
    "Train: RAVDESS+CREMA -> Test: TESS",
    "cm_rav_cre_to_tess.png"
)
# =========================================================
# TASK 3:
# POOLING EXPERIMENTS
# =========================================================
print("\nPooling Study\n")

pooling_methods = [
    "mean",
    "max",
    "std",
    "mean+std"
]

results = []

for pool_type in pooling_methods:

    print(f"POOLING METHOD: {pool_type.upper()}")

    # LOAD DATA AGAIN WITH NEW POOLING
    X_rav_pool, y_rav_pool, _ = load_ravdess(
        RAVDESS_PATH,
        pooling=pool_type
    )

    X_cre_pool, y_cre_pool, _ = load_crema(
        CREMA_PATH,
        pooling=pool_type
    )

    X_pool = np.vstack((
        X_rav_pool,
        X_cre_pool
    ))

    y_pool = np.hstack((
        y_rav_pool,
        y_cre_pool
    ))

    # TRAIN TEST SPLIT
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_pool,
        y_pool,
        test_size=0.2,
        stratify=y_pool,
        random_state=42
    )

    # SCALE
    sc = StandardScaler()

    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    # MODEL
    pool_model = LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=10000
    )

    pool_model.fit(X_tr, y_tr)

    # PREDICT
    preds = pool_model.predict(X_te)

    # METRICS
    acc = accuracy_score(y_te, preds)

    macro_f1 = f1_score(
        y_te,
        preds,
        average="macro"
    )

    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print("\n")

    results.append([
        pool_type,
        acc,
        macro_f1
    ])

    # CONFUSION MATRIX
    plot_confusion_matrix(
        y_te,
        preds,
        f"Pooling: {pool_type.upper()}",
        f"cm_pool_{pool_type}.png"
    )


#final pooling results
print("FINAL POOLING RESULTS")

for r in results:

    print(
        f"{r[0]:12s} | "
        f"Accuracy: {r[1]:.4f} | "
        f"Macro-F1: {r[2]:.4f}"
    )


print("\nDone.")

import joblib

# Add this to the very bottom of main.py
print("Saving model and scaler...")
joblib.dump(model, 'emotion_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Saved successfully!")