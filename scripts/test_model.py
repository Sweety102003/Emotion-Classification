"""
Test Script for Emotion Classification Model
This script allows testing the trained model with custom audio files
"""

import os
import sys
import numpy as np
import librosa
import joblib
import tensorflow as tf
from pathlib import Path

# Add the parent directory to the path to import the main script functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.final_emotion_classification import extract_features

def load_model_and_preprocessors():
    """Load the trained model and preprocessing objects"""
    # Use correct paths relative to project root
    model_path = 'models/final_emotion_classifier_model.h5'  # Changed from 'models/...'
    scaler_path = 'models/final_scaler.pkl'  # Changed from 'models/...'
    label_encoder_path = 'models/final_label_encoder.pkl'  # Changed from 'models/...'
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label encoder file not found: {label_encoder_path}")
    
    # Load the model and preprocessing objects
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    
    return model, scaler, label_encoder

def predict_emotion_from_file(audio_file_path, model, scaler, label_encoder):
    """Predict emotion from an audio file"""
    try:
        # Extract features
        features = extract_features(audio_file_path)
        if features is None:
            return {
                'error': f"Could not extract features from {audio_file_path}",
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
        
        return {
            'emotion': predicted_emotion,
            'confidence': confidence,
            'probabilities': probabilities,
            'error': None
        }
    except Exception as e:
        return {
            'error': f"Error processing {audio_file_path}: {str(e)}",
            'emotion': None,
            'confidence': 0.0,
            'probabilities': {}
        }

def test_single_file(audio_file_path):
    """Test a single audio file"""
    print(f"Testing file: {audio_file_path}")
    print("-" * 50)
    
    # Load model and preprocessors
    try:
        model, scaler, label_encoder = load_model_and_preprocessors()
        print("✓ Model and preprocessors loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Predict emotion
    result = predict_emotion_from_file(audio_file_path, model, scaler, label_encoder)
    
    if result['error']:
        print(f"✗ {result['error']}")
        return
    
    # Display results
    print(f"Predicted Emotion: {result['emotion']}")
    print(f"Confidence: {result['confidence']:.4f} ({result['confidence']:.2%})")
    print("\nProbabilities for all emotions:")
    for emotion, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {emotion}: {prob:.4f} ({prob:.2%})")
    
    print("-" * 50)

def test_directory(directory_path):
    """Test all audio files in a directory"""
    print(f"Testing all audio files in: {directory_path}")
    print("=" * 60)
    
    # Load model and preprocessors
    try:
        model, scaler, label_encoder = load_model_and_preprocessors()
        print("✓ Model and preprocessors loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Find all audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(Path(directory_path).glob(f"*{ext}"))
        audio_files.extend(Path(directory_path).glob(f"*{ext.upper()}"))
    
    if not audio_files:
        print(f"No audio files found in {directory_path}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    print()
    
    # Test each file
    results = []
    for audio_file in audio_files:
        print(f"Testing: {audio_file.name}")
        result = predict_emotion_from_file(str(audio_file), model, scaler, label_encoder)
        
        if result['error']:
            print(f"  ✗ Error: {result['error']}")
        else:
            print(f"  ✓ Emotion: {result['emotion']} (Confidence: {result['confidence']:.2%})")
            results.append(result)
        
        print()
    
    # Summary
    if results:
        print("Summary:")
        print("-" * 30)
        emotions = [r['emotion'] for r in results]
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        for emotion, count in sorted(emotion_counts.items()):
            percentage = (count / len(results)) * 100
            print(f"{emotion}: {count} files ({percentage:.1f}%)")
        
        avg_confidence = np.mean([r['confidence'] for r in results])
        print(f"\nAverage confidence: {avg_confidence:.2%}")

def main():
    """Main function"""
    print("=== Emotion Classification Model Test Script ===\n")
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_model.py <audio_file_path>")
        print("  python test_model.py --dir <directory_path>")
        print("\nExamples:")
        print("  python test_model.py data/Actor_01/03-01-03-02-01-01-01.wav")
        print("  python test_model.py --dir test_audio/")
        return
    
    if sys.argv[1] == "--dir" and len(sys.argv) >= 3:
        directory_path = sys.argv[2]
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}")
            return
        test_directory(directory_path)
    else:
        audio_file_path = sys.argv[1]
        if not os.path.exists(audio_file_path):
            print(f"File not found: {audio_file_path}")
            return
        test_single_file(audio_file_path)

if __name__ == "__main__":
    main() 