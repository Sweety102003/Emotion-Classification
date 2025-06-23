"""
Streamlit Web App for Emotion Classification
A web interface for uploading audio files and predicting emotions
"""

import streamlit as st
import os
import sys
import numpy as np
import librosa
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tempfile
import io
import base64

# Add the parent directory to the path to import the main script functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the feature extraction function
try:
    from scripts.final_emotion_classification import extract_features
except ImportError:
    st.error("Could not import feature extraction function. Please ensure the scripts directory is available.")
    st.stop()

# Set page config
st.set_page_config(
    page_title="Emotion Classification from Speech",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .emotion-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 1rem;
        border: 2px dashed #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Emotion colors for visualization
EMOTION_COLORS = {
    'happy': '#ff7f0e',
    'sad': '#2ca02c',
    'angry': '#d62728',
    'fearful': '#9467bd',
    'disgust': '#8c564b',
    'surprised': '#e377c2',
    'neutral': '#7f7f7f',
    'calm': '#bcbd22'
}

def load_model_and_preprocessors():
    """Load the trained model and preprocessing objects"""
    model_path = '../models/final_emotion_classifier_model.h5'
    scaler_path = '../models/final_scaler.pkl'
    label_encoder_path = '../models/final_label_encoder.pkl'
    
    # Check if files exist
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None, None, None
    if not os.path.exists(scaler_path):
        st.error(f"Scaler file not found: {scaler_path}")
        return None, None, None
    if not os.path.exists(label_encoder_path):
        st.error(f"Label encoder file not found: {label_encoder_path}")
        return None, None, None
    
    try:
        # Load the model and preprocessing objects
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(label_encoder_path)
        return model, scaler, label_encoder
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def predict_emotion_from_audio(audio_file, model, scaler, label_encoder):
    """Predict emotion from uploaded audio file"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name
        
        # Extract features
        features = extract_features(tmp_file_path)
        if features is None:
            return {
                'error': "Could not extract features from audio file",
                'emotion': None,
                'confidence': 0.0,
                'probabilities': {}
            }
        
        # Preprocess features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        prediction = model.predict(features_scaled, verbose=0)
        predicted_class = np.argmax(prediction[0])
        predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]
        confidence = np.max(prediction[0])
        
        # Get probabilities for all emotions
        probabilities = dict(zip(label_encoder.classes_, prediction[0]))
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return {
            'emotion': predicted_emotion,
            'confidence': confidence,
            'probabilities': probabilities,
            'error': None
        }
    except Exception as e:
        return {
            'error': f"Error processing audio file: {str(e)}",
            'emotion': None,
            'confidence': 0.0,
            'probabilities': {}
        }

def create_emotion_visualization(probabilities):
    """Create a bar chart visualization of emotion probabilities"""
    emotions = list(probabilities.keys())
    probs = list(probabilities.values())
    colors = [EMOTION_COLORS.get(emotion, '#1f77b4') for emotion in emotions]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(emotions, probs, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('Emotion Probabilities', fontsize=16, fontweight='bold')
    ax.set_xlabel('Emotion', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def create_audio_visualization(audio_file):
    """Create audio waveform and spectrogram visualization"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name
        
        # Load audio
        audio, sr = librosa.load(tmp_file_path, sr=22050)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Waveform
        ax1.plot(audio, color='#1f77b4', alpha=0.7)
        ax1.set_title('Audio Waveform', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax2)
        ax2.set_title('Spectrogram', fontsize=14, fontweight='bold')
        fig.colorbar(img, ax=ax2, format='%+2.0f dB')
        
        plt.tight_layout()
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return fig
    except Exception as e:
        st.error(f"Error creating audio visualization: {e}")
        return None

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🎤 Emotion Classification from Speech</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Upload an audio file to classify the emotion expressed in the speech
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h3 class="sub-header">📊 About</h3>', unsafe_allow_html=True)
        st.markdown("""
        This application uses a deep learning model trained on the RAVDESS dataset 
        to classify emotions from speech audio.
        
        **Supported Emotions:**
        - 😊 Happy
        - 😢 Sad  
        - 😠 Angry
        - 😨 Fearful
        - 🤢 Disgust
        - 😲 Surprised
        - 😐 Neutral
        - 😌 Calm
        
        **Supported Audio Formats:**
        - WAV
        - MP3
        - FLAC
        - M4A
        - OGG
        """)
        
        st.markdown('<h3 class="sub-header">📈 Model Performance</h3>', unsafe_allow_html=True)
        st.metric("Overall Accuracy", "68.06%")
        st.metric("F1 Score", "67.23%")
        st.metric("Classes", "8 emotions")
        
        st.markdown('<h3 class="sub-header">🔧 Technical Details</h3>', unsafe_allow_html=True)
        st.markdown("""
        - **Model**: Deep Neural Network
        - **Features**: MFCC, Chroma, Mel, Spectral
        - **Dataset**: RAVDESS (1440 samples)
        - **Training**: 200 epochs with early stopping
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">🎵 Upload Audio File</h2>', unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
            help="Upload an audio file containing speech to classify the emotion"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.1f} KB",
                "File type": uploaded_file.type
            }
            
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            st.write("**File Information:**")
            for key, value in file_details.items():
                st.write(f"- {key}: {value}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Load model
            with st.spinner("Loading model..."):
                model, scaler, label_encoder = load_model_and_preprocessors()
            
            if model is not None:
                # Make prediction
                with st.spinner("Analyzing audio..."):
                    result = predict_emotion_from_audio(uploaded_file, model, scaler, label_encoder)
                
                if result['error']:
                    st.error(result['error'])
                else:
                    # Display results
                    st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
                    
                    # Main prediction
                    col_pred1, col_pred2 = st.columns([1, 1])
                    
                    with col_pred1:
                        st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
                        st.markdown(f"**Predicted Emotion:** {result['emotion'].upper()}")
                        st.markdown(f"**Confidence:** {result['confidence']:.2%}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col_pred2:
                        # Emotion emoji mapping
                        emotion_emoji = {
                            'happy': '😊',
                            'sad': '😢',
                            'angry': '😠',
                            'fearful': '😨',
                            'disgust': '🤢',
                            'surprised': '😲',
                            'neutral': '😐',
                            'calm': '😌'
                        }
                        
                        emoji = emotion_emoji.get(result['emotion'], '🎭')
                        st.markdown(f'<div style="text-align: center; font-size: 4rem;">{emoji}</div>', unsafe_allow_html=True)
                    
                    # Probability visualization
                    st.markdown('<h3 class="sub-header">📊 Emotion Probabilities</h3>', unsafe_allow_html=True)
                    fig = create_emotion_visualization(result['probabilities'])
                    st.pyplot(fig)
                    
                    # Detailed probabilities
                    st.markdown('<h3 class="sub-header">📋 Detailed Results</h3>', unsafe_allow_html=True)
                    
                    # Sort probabilities
                    sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
                    
                    for i, (emotion, prob) in enumerate(sorted_probs):
                        emoji = emotion_emoji.get(emotion, '🎭')
                        color = EMOTION_COLORS.get(emotion, '#1f77b4')
                        
                        # Create progress bar
                        st.markdown(f"**{emoji} {emotion.capitalize()}**")
                        st.progress(prob)
                        st.markdown(f"<div style='text-align: right; color: {color}; font-weight: bold;'>{prob:.2%}</div>", unsafe_allow_html=True)
                        st.markdown("---")
    
    with col2:
        st.markdown('<h2 class="sub-header">🎵 Audio Preview</h2>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Audio player
            st.audio(uploaded_file, format='audio/wav')
            
            # Audio visualization
            st.markdown('<h3 class="sub-header">📈 Audio Analysis</h3>', unsafe_allow_html=True)
            
            # Reset file pointer for visualization
            uploaded_file.seek(0)
            fig_audio = create_audio_visualization(uploaded_file)
            
            if fig_audio:
                st.pyplot(fig_audio)
        else:
            st.info("Upload an audio file to see the preview and analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>Built with Streamlit • Powered by TensorFlow • RAVDESS Dataset</p>
        <p>Emotion Classification from Speech Audio</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 