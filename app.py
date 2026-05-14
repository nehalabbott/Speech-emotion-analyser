import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
import os

# --- 1. Load the Model and Scaler ---
@st.cache_resource # This keeps the model loaded in memory so it's fast
def load_model():
    model = joblib.load('emotion_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# --- 2. Your Exact Extraction Functions ---
def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    delta_mfcc = librosa.feature.delta(mfcc).mean(axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)
    return np.hstack((mfcc_mean, mfcc_std, delta_mfcc, chroma, mel, rms))

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
    # Display an audio player so you can hear what you uploaded
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Analyze Emotion"):
        with st.spinner("Extracting vocal features and analyzing..."):
            
            # Streamlit uploads files into memory. We need to save it to a temp file 
            # temporarily so librosa can read it properly.
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                temp_audio.write(uploaded_file.read())
                temp_filepath = temp_audio.name
            
            # Process the file using your functions
            audio, sr = preprocess(temp_filepath)
            
            if audio is not None:
                features = extract_features(audio, sr)
                
                # Scale the features
                features_scaled = scaler.transform([features])
                
                # Predict
                prediction = model.predict(features_scaled)[0]
                
                # Make it look nice
                emotion_emojis = {
                    "angry": "😡 Angry", "happy": "😄 Happy", "sad": "😢 Sad", 
                    "neutral": "😐 Neutral", "fear": "😨 Fear", "disgust": "🤢 Disgust", 
                    "surprise": "😲 Surprise"
                }
                
                display_text = emotion_emojis.get(prediction, prediction.capitalize())
                
                st.success("Analysis Complete!")
                st.metric(label="Detected Emotion", value=display_text)
            
            # Clean up the temp file
            os.remove(temp_filepath)


