import os
import warnings
warnings.filterwarnings("ignore")

import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from dotenv import load_dotenv

from sklearn.model_selection import (
    GroupShuffleSplit
)

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix
)

# =========================================================
# LOAD ENV VARIABLES
# =========================================================
load_dotenv()

RAVDESS_PATH = os.getenv("RAVDESS_PATH")
CREMA_PATH = os.getenv("CREMA_PATH")
TESS_PATH = os.getenv("TESS_PATH")

# =========================================================
# COMMON LABELS
# =========================================================
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

        target_length = duration * sr

        # FIX LENGTH
        if len(audio) < target_length:

            audio = np.pad(
                audio,
                (0, target_length - len(audio))
            )

        else:
            audio = audio[:target_length]

        # NORMALIZE
        audio = librosa.util.normalize(audio)

        return audio, sr

    except Exception as e:

        print(f"Error processing {file_path}: {e}")

        return None, None


# =========================================================
# FEATURE EXTRACTION
# =========================================================
def extract_raw_sequences(audio, sr):

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    # DELTA
    delta_mfcc = librosa.feature.delta(mfcc)

    # DELTA-DELTA
    delta2_mfcc = librosa.feature.delta(
        mfcc,
        order=2
    )

    # CHROMA
    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        n_chroma=12
    )

    # LOG MEL
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=audio,
            sr=sr
        ),
        ref=np.max
    )

    # RMS
    rms = librosa.feature.rms(y=audio)

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(audio)

    # SPECTRAL CONTRAST
    spectral_contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=sr
    )

    # STACK FEATURES
    return np.vstack((
        mfcc,
        delta_mfcc,
        delta2_mfcc,
        chroma,
        mel,
        rms,
        zcr,
        spectral_contrast
    ))


# =========================================================
# POOLING
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
# LABEL FUNCTIONS
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


def get_label_tess(filename):

    emotion = filename.split("_")[-1].split(".")[0]

    mapping = {
        "angry": "angry",
        "happy": "happy",
        "sad": "sad",
        "neutral": "neutral",
        "fear": "fear",
        "disgust": "disgust",
        "ps": None
    }

    return mapping.get(emotion)


def get_speaker_tess(filename):

    return filename.split("_")[0]


# =========================================================
# RAW FEATURE LOADER
# =========================================================
def load_raw_sequences(path, dataset_type):

    X_raw = []
    y = []
    groups = []

    for root, _, files in os.walk(path):

        for file in files:

            if not file.endswith(".wav"):
                continue

            # LABELS
            if dataset_type == "ravdess":

                label = get_label_ravdess(file)
                speaker = get_speaker_ravdess(file)

            elif dataset_type == "crema":

                label = get_label_crema(file)
                speaker = get_speaker_crema(file)

            elif dataset_type == "tess":

                label = get_label_tess(file)
                speaker = get_speaker_tess(file)

            else:
                continue

            if label not in COMMON_EMOTIONS:
                continue

            # AUDIO
            full_path = os.path.join(root, file)

            audio, sr = preprocess(full_path)

            if audio is None:
                continue

            # FEATURES
            raw_seq = extract_raw_sequences(audio, sr)

            X_raw.append(raw_seq)
            y.append(label)
            groups.append(speaker)

    return (
        X_raw,
        np.array(y),
        np.array(groups)
    )


# =========================================================
# DATASET LOADER
# =========================================================
def load_dataset(raw_sequences, labels, groups, pooling):

    X = []

    for seq in raw_sequences:

        pooled = apply_pooling(seq, pooling)

        X.append(pooled)

    return (
        np.array(X),
        labels,
        groups
    )


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

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.savefig(
        filename,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved confusion matrix: {filename}")


# =========================================================
# EVALUATION
# =========================================================
def evaluate_model(
    model,
    X_test,
    y_test,
    experiment_name,
    cm_filename
):

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    print(experiment_name)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print()

    plot_confusion_matrix(
        y_test,
        y_pred,
        experiment_name,
        cm_filename
    )

    return accuracy, macro_f1


# =========================================================
# PRECOMPUTE FEATURES
# =========================================================
print("\nPrecomputing raw feature sequences...\n")

X_rav_raw, y_rav, g_rav = load_raw_sequences(
    RAVDESS_PATH,
    "ravdess"
)

X_cre_raw, y_cre, g_cre = load_raw_sequences(
    CREMA_PATH,
    "crema"
)

X_tess_raw, y_tess, g_tess = load_raw_sequences(
    TESS_PATH,
    "tess"
)

# =========================================================
# APPLY DEFAULT POOLING
# =========================================================
X_rav, y_rav, g_rav = load_dataset(
    X_rav_raw,
    y_rav,
    g_rav,
    pooling="mean+std"
)

X_cre, y_cre, g_cre = load_dataset(
    X_cre_raw,
    y_cre,
    g_cre,
    pooling="mean+std"
)

X_tess, y_tess, g_tess = load_dataset(
    X_tess_raw,
    y_tess,
    g_tess,
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
print()

# =========================================================
# TASK 1
# MIXED DATASET SPEAKER-INDEPENDENT
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

# SCALE
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MODEL
model = LinearSVC(
    class_weight="balanced",
    random_state=42,
    max_iter=10000
)

model.fit(X_train, y_train)

# SAVE
print("Saving main model and scaler...")

joblib.dump(model, "emotion_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Saved successfully!\n")

# TRAIN METRICS
y_pred_train = model.predict(X_train)

train_acc = accuracy_score(
    y_train,
    y_pred_train
)

train_f1 = f1_score(
    y_train,
    y_pred_train,
    average="macro"
)

print("TRAIN RESULTS")
print(f"Train Accuracy : {train_acc:.4f}")
print(f"Train Macro F1 : {train_f1:.4f}")
print()

# TEST METRICS
print("TEST RESULTS")

evaluate_model(
    model,
    X_test,
    y_test,
    "Mixed Dataset Speaker-Independent Split",
    "cm_mixed.png"
)

# =========================================================
# SAME DATASET EVALUATION
# =========================================================
def same_dataset_evaluation(
    X,
    y,
    groups,
    dataset_name,
    cm_filename
):

    print("\n" + "="*60)
    print(f"{dataset_name} SAME-DATASET EVALUATION")
    print("="*60)

    groups = np.array(groups)

    gss = GroupShuffleSplit(
        test_size=0.2,
        n_splits=1,
        random_state=42
    )

    train_idx, test_idx = next(
        gss.split(X, y, groups)
    )

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # SCALE
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # MODEL
    model = LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=10000
    )

    model.fit(X_train, y_train)

    # TRAIN
    y_pred_train = model.predict(X_train)

    train_acc = accuracy_score(
        y_train,
        y_pred_train
    )

    train_f1 = f1_score(
        y_train,
        y_pred_train,
        average="macro"
    )

    print("TRAIN RESULTS")
    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Train Macro F1 : {train_f1:.4f}")
    print()

    # TEST
    y_pred_test = model.predict(X_test)

    test_acc = accuracy_score(
        y_test,
        y_pred_test
    )

    test_f1 = f1_score(
        y_test,
        y_pred_test,
        average="macro"
    )

    print("TEST RESULTS")
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test Macro F1 : {test_f1:.4f}")
    print()

    plot_confusion_matrix(
        y_test,
        y_pred_test,
        f"{dataset_name} Same Dataset Split",
        cm_filename
    )


print("\nSame Dataset Evaluation\n")

same_dataset_evaluation(
    X_rav,
    y_rav,
    g_rav,
    "RAVDESS",
    "cm_rav_same.png"
)

same_dataset_evaluation(
    X_cre,
    y_cre,
    g_cre,
    "CREMA-D",
    "cm_cre_same.png"
)

same_dataset_evaluation(
    X_tess,
    y_tess,
    g_tess,
    "TESS",
    "cm_tess_same.png"
)

# =========================================================
# TASK 2
# CROSS DATASET
# =========================================================
print("\nStarting Task 2: Cross Dataset Generalization\n")

experiments = [

    (
        "Train: CREMA+TESS -> Test: RAVDESS",
        np.vstack((X_cre, X_tess)),
        np.hstack((y_cre, y_tess)),
        X_rav,
        y_rav,
        "cm_cre_tess_to_rav.png"
    ),

    (
        "Train: RAVDESS+TESS -> Test: CREMA",
        np.vstack((X_rav, X_tess)),
        np.hstack((y_rav, y_tess)),
        X_cre,
        y_cre,
        "cm_rav_tess_to_cre.png"
    ),

    (
        "Train: RAVDESS+CREMA -> Test: TESS",
        np.vstack((X_rav, X_cre)),
        np.hstack((y_rav, y_cre)),
        X_tess,
        y_tess,
        "cm_rav_cre_to_tess.png"
    )
]

for (
    title,
    X_train,
    y_train,
    X_test,
    y_test,
    cm_name
) in experiments:

    print("\n" + "="*60)
    print(title)
    print("="*60)

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
        title,
        cm_name
    )

# =========================================================
# TASK 3
# POOLING STUDY
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

    # APPLY POOLING
    X_rav_pool = np.array([
        apply_pooling(seq, pool_type)
        for seq in X_rav_raw
    ])

    X_cre_pool = np.array([
        apply_pooling(seq, pool_type)
        for seq in X_cre_raw
    ])

    X_pool = np.vstack((
        X_rav_pool,
        X_cre_pool
    ))

    y_pool = np.hstack((
        y_rav,
        y_cre
    ))

    groups_pool = np.hstack((
        ["rav_" + g for g in g_rav],
        ["cre_" + g for g in g_cre]
    ))

    # SPEAKER-INDEPENDENT SPLIT
    gss = GroupShuffleSplit(
        test_size=0.2,
        n_splits=1,
        random_state=42
    )

    train_idx, test_idx = next(
        gss.split(
            X_pool,
            y_pool,
            groups_pool
        )
    )

    X_tr = X_pool[train_idx]
    X_te = X_pool[test_idx]

    y_tr = y_pool[train_idx]
    y_te = y_pool[test_idx]

    # SCALE
    scaler = StandardScaler()

    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # MODEL
    model = LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=10000
    )

    model.fit(X_tr, y_tr)

    # PREDICT
    preds = model.predict(X_te)

    # METRICS
    acc = accuracy_score(y_te, preds)

    macro_f1 = f1_score(
        y_te,
        preds,
        average="macro"
    )

    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print()

    results.append([
        pool_type,
        acc,
        macro_f1
    ])

    plot_confusion_matrix(
        y_te,
        preds,
        f"Pooling: {pool_type.upper()}",
        f"cm_pool_{pool_type}.png"
    )

# =========================================================
# FINAL RESULTS
# =========================================================
print("FINAL POOLING RESULTS")

for r in results:

    print(
        f"{r[0]:12s} | "
        f"Accuracy: {r[1]:.4f} | "
        f"Macro-F1: {r[2]:.4f}"
    )

print("\nDone.")