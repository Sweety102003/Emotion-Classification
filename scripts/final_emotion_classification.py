

import os
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
import random
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
    if len(parts) >= 7:
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
    else:
        print(f"[WARN] Skipping file with unexpected format: {filename} (parts: {parts})")
    return None

def augment_audio(audio, sr):
    """Apply random augmentation to audio."""
    # Random noise
    if random.random() < 0.3:
        noise = np.random.normal(0, 0.005, len(audio))
        audio = audio + noise
    # Random pitch shift
    if random.random() < 0.3:
        steps = float(np.random.uniform(-2, 2))
        audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=steps)
    # Random time stretch
    if random.random() < 0.3:
        rate = float(np.random.uniform(0.8, 1.2))
        audio = librosa.effects.time_stretch(y=audio, rate=rate)
        # Pad or trim to original length
        target_len = int(sr * 3)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), 'constant')
        else:
            audio = audio[:target_len]
    return audio

def extract_features_from_audio(audio, sr):
    """Extract features from a numpy audio array and sample rate."""
    try:
        # Pad or truncate to ensure consistent length
        max_length = 3
        target_len = int(sr * max_length)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), 'constant')
        else:
            audio = audio[:target_len]
        # Extract features (same as extract_features)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        mfcc_std = np.std(mfcc.T, axis=0)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_scaled = np.mean(chroma.T, axis=0)
        chroma_std = np.std(chroma.T, axis=0)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_scaled = np.mean(mel.T, axis=0)
        mel_std = np.std(mel.T, axis=0)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_scaled = np.mean(contrast.T, axis=0)
        contrast_std = np.std(contrast.T, axis=0)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        tonnetz_scaled = np.mean(tonnetz.T, axis=0)
        tonnetz_std = np.std(tonnetz.T, axis=0)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        centroid_scaled = np.mean(centroid.T, axis=0)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        rolloff_scaled = np.mean(rolloff.T, axis=0)
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_scaled = np.mean(zcr.T, axis=0)
        rms = librosa.feature.rms(y=audio)
        rms_scaled = np.mean(rms.T, axis=0)
        features = np.concatenate([
            mfcc_scaled, mfcc_std,
            chroma_scaled, chroma_std,
            mel_scaled[:20], mel_std[:20],
            contrast_scaled, contrast_std,
            tonnetz_scaled, tonnetz_std,
            centroid_scaled, rolloff_scaled, zcr_scaled, rms_scaled
        ])
        return features
    except Exception as e:
        print(f"Error processing audio array: {e}")
        return None

def extract_features(file_path, max_length=3):
    """Extract audio features from a single file"""
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=22050, duration=max_length)
        
        # Pad or truncate to ensure consistent length
        target_len = int(sr * max_length)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), 'constant')
        else:
            audio = audio[:target_len]
        
        # Extract features
        # MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        mfcc_std = np.std(mfcc.T, axis=0)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_scaled = np.mean(chroma.T, axis=0)
        chroma_std = np.std(chroma.T, axis=0)
        
        # Mel spectrogram features
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_scaled = np.mean(mel.T, axis=0)
        mel_std = np.std(mel.T, axis=0)
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_scaled = np.mean(contrast.T, axis=0)
        contrast_std = np.std(contrast.T, axis=0)
        
        # Tonnetz features
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        tonnetz_scaled = np.mean(tonnetz.T, axis=0)
        tonnetz_std = np.std(tonnetz.T, axis=0)
        
        # Additional features
        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        centroid_scaled = np.mean(centroid.T, axis=0)
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        rolloff_scaled = np.mean(rolloff.T, axis=0)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_scaled = np.mean(zcr.T, axis=0)
        
        # Root Mean Square Energy
        rms = librosa.feature.rms(y=audio)
        rms_scaled = np.mean(rms.T, axis=0)
        
        # Combine all features
        features = np.concatenate([
            mfcc_scaled, mfcc_std,
            chroma_scaled, chroma_std,
            mel_scaled[:20], mel_std[:20],  # Reduced mel features
            contrast_scaled, contrast_std,
            tonnetz_scaled, tonnetz_std,
            centroid_scaled, rolloff_scaled, zcr_scaled, rms_scaled
        ])
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def create_improved_model(input_shape, num_classes):
    """Create an improved deep learning model for emotion classification"""
    model = tf.keras.Sequential([
        # Input layer with batch normalization
        tf.keras.layers.Dense(512, activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        # Hidden layers
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model with better optimizer and learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def predict_emotion(audio_file_path):
    """Predict emotion from an audio file"""
    # Load the saved model and preprocessing objects
    model = tf.keras.models.load_model(f'{MODEL_PATH}/final_emotion_classifier_model.h5')
    scaler = joblib.load(f'{MODEL_PATH}/final_scaler.pkl')
    label_encoder = joblib.load(f'{MODEL_PATH}/final_label_encoder.pkl')
    
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
    print("=== Final Emotion Classification Pipeline ===\n")
    
    # 1. Load and parse data
    print("1. Loading and parsing audio files...")
    audio_files = []
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
                            print(f"[WARN] File skipped due to parsing issue: {file}")
    
    df = pd.DataFrame(audio_files)
    print(f"Total audio files found: {len(df)}")
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
            # Data augmentation: only for training data (first 70%)
            if idx < int(0.7 * len(df)):
                audio, sr = librosa.load(row['filepath'], sr=22050, duration=3)
                audio_aug = augment_audio(audio, sr)
                features_aug = extract_features_from_audio(audio_aug, sr)
                if features_aug is not None:
                    features_list.append(features_aug)
                    valid_indices.append(idx)

    df_filtered = df.iloc[valid_indices].reset_index(drop=True)
    features_array = np.array(features_list)
    print(f"Successfully extracted features from {len(features_array)} files")
    print(f"Feature shape: {features_array.shape}")
    
    # 3. Preprocess data
    print("\n3. Preprocessing data...")
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df_filtered['emotion'])

    print("Label encoding:")
    for i, emotion in enumerate(label_encoder.classes_):
        print(f"{i}: {emotion}")

    # Split the data with stratification
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
    model = create_improved_model(input_shape, num_classes)
    
    # Training parameters
    EPOCHS = 200
    BATCH_SIZE = 16
    PATIENCE = 25

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
        patience=10,
        min_lr=1e-7,
        verbose=1
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'{MODEL_PATH}/final_emotion_model.h5',
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
    best_model = tf.keras.models.load_model(f'{MODEL_PATH}/final_emotion_model.h5')

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
    best_model.save(f'{MODEL_PATH}/final_emotion_classifier_model.h5')
    joblib.dump(scaler, f'{MODEL_PATH}/final_scaler.pkl')
    joblib.dump(label_encoder, f'{MODEL_PATH}/final_label_encoder.pkl')

    print("Model and preprocessing objects saved successfully!")
    
    # 7. Create confusion matrix visualization
    print("\n7. Creating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix - Final Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{MODEL_PATH}/confusion_matrix_final.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8. Test prediction function
    print("\n8. Testing prediction function...")
    if len(df_filtered) > 0:
        sample_file = df_filtered.iloc[0]['filepath']
        print(f"Testing with sample file: {sample_file}")
        result = predict_emotion(sample_file)
        print(f"Predicted emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Actual emotion: {df_filtered.iloc[0]['emotion']}")
    
    print("\n=== Final Pipeline completed successfully! ===")

if __name__ == "__main__":
    main() 