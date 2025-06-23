"""
Demo Script for Emotion Classification
This script demonstrates how to use the trained model for emotion classification
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.final_emotion_classification import predict_emotion

def demo_single_prediction():
    """Demonstrate single audio file prediction"""
    print("=== Single Audio File Prediction Demo ===\n")
    
    # Find a sample audio file - use correct path
    data_path = 'data'  # Changed from '../data' to 'data'
    sample_file = None
    
    if not os.path.exists(data_path):
        print(f"Data directory not found: {data_path}")
        print("Please ensure the RAVDESS dataset is downloaded and extracted to the 'data' directory")
        return
    
    for actor_dir in os.listdir(data_path):
        if actor_dir.startswith('Actor_'):
            actor_path = os.path.join(data_path, actor_dir)
            if os.path.isdir(actor_path):
                for file in os.listdir(actor_path):
                    if file.endswith('.wav'):
                        sample_file = os.path.join(actor_path, file)
                        break
                if sample_file:
                    break
    
    if not sample_file:
        print("No sample audio files found!")
        print("Please ensure the RAVDESS dataset is properly downloaded and extracted")
        return
    
    print(f"Using sample file: {sample_file}")
    print("-" * 50)
    
    # Make prediction
    result = predict_emotion(sample_file)
    
    if isinstance(result, str):
        print(f"Error: {result}")
        return
    
    # Display results
    print(f"Predicted Emotion: {result['emotion'].upper()}")
    print(f"Confidence: {result['confidence']:.2%}")
    
    print("\nProbabilities for all emotions:")
    sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
    
    for emotion, prob in sorted_probs:
        print(f"  {emotion.capitalize()}: {prob:.2%}")
    
    # Create visualization
    create_probability_plot(result['probabilities'])

def demo_multiple_predictions():
    """Demonstrate multiple audio file predictions"""
    print("\n=== Multiple Audio Files Prediction Demo ===\n")
    
    data_path = 'data'  # Changed from '../data' to 'data'
    results = []
    
    if not os.path.exists(data_path):
        print(f"Data directory not found: {data_path}")
        return
    
    # Process a few files from different actors
    count = 0
    max_files = 5
    
    for actor_dir in os.listdir(data_path):
        if actor_dir.startswith('Actor_') and count < max_files:
            actor_path = os.path.join(data_path, actor_dir)
            if os.path.isdir(actor_path):
                for file in os.listdir(actor_path):
                    if file.endswith('.wav') and count < max_files:
                        file_path = os.path.join(actor_path, file)
                        print(f"Processing: {file}")
                        
                        result = predict_emotion(file_path)
                        if not isinstance(result, str):
                            results.append({
                                'file': file,
                                'emotion': result['emotion'],
                                'confidence': result['confidence']
                            })
                        
                        count += 1
                        break
    
    # Display summary
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['file']}")
        print(f"   Emotion: {result['emotion'].upper()}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print()
    
    # Emotion distribution
    emotions = [r['emotion'] for r in results]
    emotion_counts = {}
    for emotion in emotions:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    print("Emotion Distribution:")
    for emotion, count in emotion_counts.items():
        percentage = (count / len(results)) * 100
        print(f"  {emotion.capitalize()}: {count} ({percentage:.1f}%)")

def create_probability_plot(probabilities):
    """Create a bar plot of emotion probabilities"""
    emotions = list(probabilities.keys())
    probs = list(probabilities.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(emotions, probs, color='skyblue', alpha=0.7)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Emotion Probabilities', fontsize=16, fontweight='bold')
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def show_model_info():
    """Display information about the trained model"""
    print("=== Model Information ===\n")
    
    # Use correct paths relative to project root
    model_path = 'models/final_emotion_classifier_model.h5'  # Changed from '../models/...'
    scaler_path = 'models/final_scaler.pkl'  # Changed from '../models/...'
    label_encoder_path = 'models/final_label_encoder.pkl'  # Changed from '../models/...'
    
    print("Model Files:")
    print(f"  Model: {'✓' if os.path.exists(model_path) else '✗'} {model_path}")
    print(f"  Scaler: {'✓' if os.path.exists(scaler_path) else '✗'} {scaler_path}")
    print(f"  Label Encoder: {'✓' if os.path.exists(label_encoder_path) else '✗'} {label_encoder_path}")
    
    print("\nModel Performance:")
    print("  Overall Accuracy: 68.06%")
    print("  F1 Score: 67.23%")
    print("  Classes: 8 emotions")
    
    print("\nSupported Emotions:")
    emotions = ['happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised', 'neutral', 'calm']
    for emotion in emotions:
        print(f"  - {emotion.capitalize()}")
    
    print("\nFeature Extraction:")
    print("  - MFCC (Mel-Frequency Cepstral Coefficients)")
    print("  - Chroma Features")
    print("  - Mel Spectrogram")
    print("  - Spectral Contrast")
    print("  - Tonnetz Features")
    print("  - Additional spectral features")
    print("  Total: 120 features per audio sample")

def main():
    """Main demo function"""
    print("🎤 Emotion Classification Demo")
    print("=" * 50)
    
    # Show model information
    show_model_info()
    
    # Demo single prediction
    demo_single_prediction()
    
    # Demo multiple predictions
    demo_multiple_predictions()
    
    print("\n" + "=" * 50)
    print("Demo completed! 🎉")
    print("\nTo use the model in your own code:")
    print("from scripts.final_emotion_classification import predict_emotion")
    print("result = predict_emotion('path/to/audio.wav')")
    print("print(f'Emotion: {result[\"emotion\"]}')")

if __name__ == "__main__":
    main() 