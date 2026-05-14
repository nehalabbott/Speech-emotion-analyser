import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
import os

# --- 1. Load the Model and Scaler ---
@st.cache_resource 
def load_model():
    model = joblib.load('emotion_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# --- 2. MATCHING FEATURE EXTRACTION (Synced with main.py) ---
def extract_features(audio, sr):
    # Extract raw sequences exactly as main.py does
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    delta_mfcc = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    rms = librosa.feature.rms(y=audio)
    zcr = librosa.feature.zero_crossing_rate(audio)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

    # Stack them
    sequence = np.vstack((
        mfcc, delta_mfcc, chroma, mel, rms, zcr, spectral_contrast
    ))

    # Apply "mean+std" pooling
    pooled = np.hstack((
        np.mean(sequence, axis=1),
        np.std(sequence, axis=1)
    ))
    return pooled

def preprocess(file_path, duration=3):
    try:
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        target_length = duration * sr
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        audio = librosa.util.normalize(audio)
        return audio, sr
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None

# --- 3. The Streamlit UI ---
st.set_page_config(page_title="Speech Emotion Analyser", page_icon="🎙️")

st.title("🎙️ Speech Emotion Analyser")
st.write("Upload a 3-second audio clip to detect the underlying emotion.")

# File Uploader
uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Analyze Emotion"):
        with st.spinner("Extracting advanced vocal features and analyzing..."):
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                temp_audio.write(uploaded_file.read())
                temp_filepath = temp_audio.name
            
            # Process the file
            audio, sr = preprocess(temp_filepath)
            
            if audio is not None:
                features = extract_features(audio, sr)
                
                # Scale the features
                features_scaled = scaler.transform([features])
                
                # Predict
                prediction = model.predict(features_scaled)[0]
                
                # Update emojis to match your COMMON_EMOTIONS list
                emotion_emojis = {
                    "angry": "😡 Angry", 
                    "happy": "😄 Happy", 
                    "sad": "😢 Sad", 
                    "neutral": "😐 Neutral", 
                    "fear": "😨 Fear", 
                    "disgust": "🤢 Disgust"
                }
                
                display_text = emotion_emojis.get(prediction, prediction.capitalize())
                
                st.success("Analysis Complete!")
                st.metric(label="Detected Emotion", value=display_text)
            
            # Clean up
            os.remove(temp_filepath)