"""
Emotion Classification Pipeline
Complete implementation for RAVDESS dataset emotion classification
"""

import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
DATA_PATH = 'data'
MODEL_PATH = 'models'

# Create models directory if it doesn't exist
os.makedirs(MODEL_PATH, exist_ok=True)

# Emotion mapping for RAVDESS dataset
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def parse_filename(filename):
    """Parse RAVDESS filename to extract emotion and other metadata"""
    parts = filename.replace('.wav', '').split('-')
    
    if len(parts) >= 7:  # RAVDESS format has 7 parts
        emotion_code = parts[2]
        intensity = parts[3]
        statement = parts[4]
        repetition = parts[5]
        actor = parts[6]
        
        return {
            'filename': filename,
            'emotion_code': emotion_code,
            'emotion': emotion_map.get(emotion_code, 'unknown'),
            'intensity': intensity,
            'statement': statement,
            'repetition': repetition,
            'actor': actor
        }
    return None

def extract_features(file_path, max_length=3):
    """Extract audio features from a single file"""
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=22050, duration=max_length)
        
        # Pad or truncate to ensure consistent length
        if len(audio) < sr * max_length:
            audio = np.pad(audio, (0, sr * max_length - len(audio)), 'constant')
        else:
            audio = audio[:sr * max_length]
        
        # Extract features
        # MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_scaled = np.mean(chroma.T, axis=0)
        
        # Mel spectrogram features
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_scaled = np.mean(mel.T, axis=0)
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_scaled = np.mean(contrast.T, axis=0)
        
        # Tonnetz features
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        tonnetz_scaled = np.mean(tonnetz.T, axis=0)
        
        # Combine all features
        features = np.concatenate([
            mfcc_scaled, chroma_scaled, mel_scaled[:20], 
            contrast_scaled, tonnetz_scaled
        ])
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def create_model(input_shape, num_classes):
    """Create a deep learning model for emotion classification"""
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Dense(256, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        
        # Hidden layers
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        
        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def predict_emotion(audio_file_path):
    """Predict emotion from an audio file"""
    # Load the saved model and preprocessing objects
    model = tf.keras.models.load_model(f'{MODEL_PATH}/emotion_classifier_model.h5')
    scaler = joblib.load(f'{MODEL_PATH}/scaler.pkl')
    label_encoder = joblib.load(f'{MODEL_PATH}/label_encoder.pkl')
    
    # Extract features
    features = extract_features(audio_file_path)
    if features is None:
        return "Error: Could not extract features from audio file"
    
    # Preprocess features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Make prediction
    prediction = model.predict(features_scaled)
    predicted_class = np.argmax(prediction[0])
    predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]
    confidence = np.max(prediction[0])
    
    return {
        'emotion': predicted_emotion,
        'confidence': confidence,
        'probabilities': dict(zip(label_encoder.classes_, prediction[0]))
    }

def main():
    """Main pipeline execution"""
    print("=== Emotion Classification Pipeline ===\n")
    
    # 1. Load and parse data
    print("1. Loading and parsing audio files...")
    audio_files = []
    skipped_files = 0
    for actor_dir in os.listdir(DATA_PATH):
        if actor_dir.startswith('Actor_'):
            actor_path = os.path.join(DATA_PATH, actor_dir)
            if os.path.isdir(actor_path):
                for file in os.listdir(actor_path):
                    if file.endswith('.wav'):
                        metadata = parse_filename(file)
                        if metadata:
                            metadata['filepath'] = os.path.join(actor_path, file)
                            audio_files.append(metadata)
                        else:
                            skipped_files += 1
    
    # Create DataFrame
    df = pd.DataFrame(audio_files)
    print(f"Total audio files found: {len(df)}")
    print(f"Files skipped due to parsing issues: {skipped_files}")
    if len(df) == 0:
        print("WARNING: No audio files were loaded. Please check the data directory and filename format.")
        # Debug: Show first few filenames and parsing results
        print("\nDebug: Checking first few filenames...")
        for actor_dir in os.listdir(DATA_PATH):
            if actor_dir.startswith('Actor_'):
                actor_path = os.path.join(DATA_PATH, actor_dir)
                if os.path.isdir(actor_path):
                    for i, file in enumerate(os.listdir(actor_path)):
                        if file.endswith('.wav') and i < 3:
                            print(f"Filename: {file}")
                            print(f"Parts: {file.replace('.wav', '').split('-')}")
                            print(f"Parsed: {parse_filename(file)}")
                            print("---")
                    break  # Only check first actor folder
    else:
        print("Dataset overview:")
        print(df['emotion'].value_counts())
    
    # 2. Extract features
    print("\n2. Extracting features from audio files...")
    features_list = []
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        features = extract_features(row['filepath'])
        if features is not None:
            features_list.append(features)
            valid_indices.append(idx)

    # Filter DataFrame to only include valid files
    df_filtered = df.iloc[valid_indices].reset_index(drop=True)
    features_array = np.array(features_list)
    print(f"Successfully extracted features from {len(features_array)} files")
    print(f"Feature shape: {features_array.shape}")
    
    # 3. Preprocess data
    print("\n3. Preprocessing data...")
    # Encode emotion labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df_filtered['emotion'])

    print("Label encoding:")
    for i, emotion in enumerate(label_encoder.classes_):
        print(f"{i}: {emotion}")

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(
        features_array, labels, test_size=0.3, random_state=42, stratify=labels
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\nData split:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Create and train model
    print("\n4. Creating and training model...")
    input_shape = (X_train_scaled.shape[1],)
    num_classes = len(label_encoder.classes_)
    model = create_model(input_shape, num_classes)
    
    # Training parameters
    EPOCHS = 100
    BATCH_SIZE = 32
    PATIENCE = 15

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'{MODEL_PATH}/best_emotion_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # Train the model
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    # 5. Evaluate model
    print("\n5. Evaluating model...")
    # Load best model
    best_model = tf.keras.models.load_model(f'{MODEL_PATH}/best_emotion_model.h5')

    # Evaluate on test set
    test_loss, test_accuracy = best_model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Predictions
    y_pred = best_model.predict(X_test_scaled)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate metrics
    overall_accuracy = accuracy_score(y_test, y_pred_classes)
    f1_macro = f1_score(y_test, y_pred_classes, average='macro')
    f1_weighted = f1_score(y_test, y_pred_classes, average='weighted')

    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")

    # Check if we meet the criteria
    print(f"\nEvaluation Criteria Check:")
    print(f"Overall Accuracy > 80%: {'✓' if overall_accuracy > 0.8 else '✗'} ({overall_accuracy:.2%})")
    print(f"F1 Score > 80%: {'✓' if f1_macro > 0.8 else '✗'} ({f1_macro:.2%})")

    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_classes, 
                              target_names=label_encoder.classes_))

    # Per-class accuracy check
    print("\nPer-class Accuracy Check:")
    class_accuracy = classification_report(y_test, y_pred_classes, 
                                         target_names=label_encoder.classes_, 
                                         output_dict=True)

    for emotion in label_encoder.classes_:
        if emotion in class_accuracy:
            acc = class_accuracy[emotion]['precision']
            status = '✓' if acc > 0.75 else '✗'
            print(f"{emotion}: {status} ({acc:.2%})")
    
    # 6. Save model and preprocessing objects
    print("\n6. Saving model and preprocessing objects...")
    best_model.save(f'{MODEL_PATH}/emotion_classifier_model.h5')
    joblib.dump(scaler, f'{MODEL_PATH}/scaler.pkl')
    joblib.dump(label_encoder, f'{MODEL_PATH}/label_encoder.pkl')

    print("Model and preprocessing objects saved successfully!")
    print(f"Model saved to: {MODEL_PATH}/emotion_classifier_model.h5")
    print(f"Scaler saved to: {MODEL_PATH}/scaler.pkl")
    print(f"Label encoder saved to: {MODEL_PATH}/label_encoder.pkl")
    
    # 7. Test prediction function
    print("\n7. Testing prediction function...")
    if len(df_filtered) > 0:
        sample_file = df_filtered.iloc[0]['filepath']
        print(f"Testing with sample file: {sample_file}")
        result = predict_emotion(sample_file)
        print(f"Predicted emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Actual emotion: {df_filtered.iloc[0]['emotion']}")
    
    print("\n=== Pipeline completed successfully! ===")

if __name__ == "__main__":
    main() 