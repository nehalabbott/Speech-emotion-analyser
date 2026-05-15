import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
import os

#load model and scaler
@st.cache_resource 
def load_model():
    model = joblib.load('emotion_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

#matching feature extraction
def extract_features(audio, sr):

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
    sequence = np.vstack((
        mfcc,
        delta_mfcc,
        delta2_mfcc,
        chroma,
        mel,
        rms,
        zcr,
        spectral_contrast
    ))

    # MEAN + STD POOLING
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

#streamlit ui
st.set_page_config(page_title="Speech Emotion Analyser", page_icon="🎙️")

st.title("🎙️ Speech Emotion Analyser")
st.write("Upload a 3-second audio clip to detect the underlying emotion.")

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Analyze Emotion"):
        with st.spinner("Extracting advanced vocal features and analyzing..."):
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                temp_audio.write(uploaded_file.read())
                temp_filepath = temp_audio.name
            
            #process the file
            audio, sr = preprocess(temp_filepath)
            
            if audio is not None:
                features = extract_features(audio, sr)
                
                #scale the features
                features_scaled = scaler.transform([features])
                
                prediction = model.predict(features_scaled)[0]
                
                emotion_emojis = {
                    "angry": "Angry", 
                    "happy": "Happy", 
                    "sad": "Sad", 
                    "neutral": "Neutral", 
                    "fear": "Fear", 
                    "disgust": "Disgust"
                }
                
                display_text = emotion_emojis.get(prediction, prediction.capitalize())
                
                st.success("Analysis Complete!")
                st.metric(label="Detected Emotion", value=display_text)
            
            # Clean up
            os.remove(temp_filepath)