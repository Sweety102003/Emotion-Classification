"""
Ensemble Emotion Classification Pipeline
Advanced implementation with ensemble methods and sophisticated feature engineering
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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
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
    return None

def extract_comprehensive_features(file_path, max_length=3):
    """Extract comprehensive audio features from a single file"""
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=22050, duration=max_length)
        
        # Pad or truncate to ensure consistent length
        if len(audio) < sr * max_length:
            audio = np.pad(audio, (0, sr * max_length - len(audio)), 'constant')
        else:
            audio = audio[:sr * max_length]
        
        # Comprehensive feature extraction
        features = []
        
        # 1. MFCC features (multiple configurations)
        mfcc_13 = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_20 = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        mfcc_40 = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        
        features.extend([np.mean(mfcc_13.T, axis=0), np.std(mfcc_13.T, axis=0)])
        features.extend([np.mean(mfcc_20.T, axis=0), np.std(mfcc_20.T, axis=0)])
        features.extend([np.mean(mfcc_40.T, axis=0), np.std(mfcc_40.T, axis=0)])
        
        # 2. Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features.extend([np.mean(chroma.T, axis=0), np.std(chroma.T, axis=0)])
        
        # 3. Mel spectrogram features
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        features.extend([np.mean(mel.T, axis=0), np.std(mel.T, axis=0)])
        
        # 4. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        
        features.extend([np.mean(spectral_centroid.T, axis=0), np.std(spectral_centroid.T, axis=0)])
        features.extend([np.mean(spectral_rolloff.T, axis=0), np.std(spectral_rolloff.T, axis=0)])
        features.extend([np.mean(spectral_bandwidth.T, axis=0), np.std(spectral_bandwidth.T, axis=0)])
        features.extend([np.mean(spectral_contrast.T, axis=0), np.std(spectral_contrast.T, axis=0)])
        
        # 5. Tonnetz features
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        features.extend([np.mean(tonnetz.T, axis=0), np.std(tonnetz.T, axis=0)])
        
        # 6. Additional features
        zcr = librosa.feature.zero_crossing_rate(audio)
        rms = librosa.feature.rms(y=audio)
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        
        features.extend([np.mean(zcr.T, axis=0), np.std(zcr.T, axis=0)])
        features.extend([np.mean(rms.T, axis=0), np.std(rms.T, axis=0)])
        features.append([tempo])
        
        # 7. Harmonic and percussive components
        harmonic, percussive = librosa.effects.hpss(audio)
        harmonic_features = librosa.feature.mfcc(y=harmonic, sr=sr, n_mfcc=13)
        percussive_features = librosa.feature.mfcc(y=percussive, sr=sr, n_mfcc=13)
        
        features.extend([np.mean(harmonic_features.T, axis=0), np.std(harmonic_features.T, axis=0)])
        features.extend([np.mean(percussive_features.T, axis=0), np.std(percussive_features.T, axis=0)])
        
        # 8. Delta and delta-delta features
        mfcc_delta = librosa.feature.delta(mfcc_13)
        mfcc_delta2 = librosa.feature.delta(mfcc_13, order=2)
        
        features.extend([np.mean(mfcc_delta.T, axis=0), np.std(mfcc_delta.T, axis=0)])
        features.extend([np.mean(mfcc_delta2.T, axis=0), np.std(mfcc_delta2.T, axis=0)])
        
        # Flatten all features
        flat_features = []
        for feature in features:
            if isinstance(feature, np.ndarray):
                flat_features.extend(feature.flatten())
            else:
                flat_features.extend(feature)
        
        return np.array(flat_features)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def create_deep_ensemble_model(input_shape, num_classes):
    """Create a deep ensemble model"""
    # Model 1: Dense Neural Network
    model1 = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model1.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model1

def create_cnn_ensemble_model(input_shape, num_classes):
    """Create a CNN ensemble model"""
    input_layer = tf.keras.layers.Input(shape=input_shape)
    reshaped = tf.keras.layers.Reshape((input_shape[0], 1))(input_layer)
    
    # Multiple CNN branches
    # Branch 1
    conv1_1 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(reshaped)
    conv1_1 = tf.keras.layers.BatchNormalization()(conv1_1)
    conv1_1 = tf.keras.layers.MaxPooling1D(2)(conv1_1)
    conv1_1 = tf.keras.layers.Dropout(0.3)(conv1_1)
    
    conv1_2 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(conv1_1)
    conv1_2 = tf.keras.layers.BatchNormalization()(conv1_2)
    conv1_2 = tf.keras.layers.GlobalAveragePooling1D()(conv1_2)
    
    # Branch 2
    conv2_1 = tf.keras.layers.Conv1D(32, 5, activation='relu', padding='same')(reshaped)
    conv2_1 = tf.keras.layers.BatchNormalization()(conv2_1)
    conv2_1 = tf.keras.layers.MaxPooling1D(2)(conv2_1)
    conv2_1 = tf.keras.layers.Dropout(0.3)(conv2_1)
    
    conv2_2 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(conv2_1)
    conv2_2 = tf.keras.layers.BatchNormalization()(conv2_2)
    conv2_2 = tf.keras.layers.GlobalAveragePooling1D()(conv2_2)
    
    # Concatenate branches
    merged = tf.keras.layers.Concatenate()([conv1_2, conv2_2])
    
    # Dense layers
    dense1 = tf.keras.layers.Dense(256, activation='relu')(merged)
    dense1 = tf.keras.layers.BatchNormalization()(dense1)
    dense1 = tf.keras.layers.Dropout(0.4)(dense1)
    
    dense2 = tf.keras.layers.Dense(128, activation='relu')(dense1)
    dense2 = tf.keras.layers.BatchNormalization()(dense2)
    dense2 = tf.keras.layers.Dropout(0.3)(dense2)
    
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(dense2)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_ensemble_models(X_train, y_train, X_val, y_val, input_shape, num_classes):
    """Train multiple models for ensemble"""
    models = []
    
    # 1. Deep Neural Network
    print("Training Deep Neural Network...")
    model1 = create_deep_ensemble_model(input_shape, num_classes)
    history1 = model1.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5)
        ],
        verbose=0
    )
    models.append(('dnn', model1))
    
    # 2. CNN Model
    print("Training CNN Model...")
    model2 = create_cnn_ensemble_model(input_shape, num_classes)
    history2 = model2.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5)
        ],
        verbose=0
    )
    models.append(('cnn', model2))
    
    # 3. Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    models.append(('rf', rf_model))
    
    # 4. Gradient Boosting
    print("Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(n_estimators=200, max_depth=10, random_state=42)
    gb_model.fit(X_train, y_train)
    models.append(('gb', gb_model))
    
    # 5. SVM
    print("Training SVM...")
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    models.append(('svm', svm_model))
    
    # 6. MLP
    print("Training MLP...")
    mlp_model = MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=500, random_state=42)
    mlp_model.fit(X_train, y_train)
    models.append(('mlp', mlp_model))
    
    return models

def ensemble_predict(models, X, label_encoder):
    """Make ensemble predictions"""
    predictions = []
    
    for name, model in models:
        if name in ['dnn', 'cnn']:
            # Deep learning models
            pred = model.predict(X)
            predictions.append(pred)
        else:
            # Traditional ML models
            pred = model.predict_proba(X)
            predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    ensemble_classes = np.argmax(ensemble_pred, axis=1)
    
    return ensemble_pred, ensemble_classes

def main():
    """Main ensemble pipeline execution"""
    print("=== Ensemble Emotion Classification Pipeline ===\n")
    
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
    
    df = pd.DataFrame(audio_files)
    print(f"Total audio files found: {len(df)}")
    print("Dataset overview:")
    print(df['emotion'].value_counts())
    
    # 2. Extract comprehensive features
    print("\n2. Extracting comprehensive features from audio files...")
    features_list = []
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        features = extract_comprehensive_features(row['filepath'])
        if features is not None:
            features_list.append(features)
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
    
    # 4. Train ensemble models
    print("\n4. Training ensemble models...")
    input_shape = (X_train_scaled.shape[1],)
    num_classes = len(label_encoder.classes_)
    
    models = train_ensemble_models(X_train_scaled, y_train, X_val_scaled, y_val, input_shape, num_classes)
    
    # 5. Evaluate ensemble model
    print("\n5. Evaluating ensemble model...")
    
    # Make ensemble predictions
    ensemble_pred, ensemble_classes = ensemble_predict(models, X_test_scaled, label_encoder)
    
    # Calculate metrics
    overall_accuracy = accuracy_score(y_test, ensemble_classes)
    f1_macro = f1_score(y_test, ensemble_classes, average='macro')
    f1_weighted = f1_score(y_test, ensemble_classes, average='weighted')

    print(f"\nEnsemble Model Results:")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")

    # Check if we meet the criteria
    print(f"\nEvaluation Criteria Check:")
    print(f"Overall Accuracy > 80%: {'✓' if overall_accuracy > 0.8 else '✗'} ({overall_accuracy:.2%})")
    print(f"F1 Score > 80%: {'✓' if f1_macro > 0.8 else '✗'} ({f1_macro:.2%})")

    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, ensemble_classes, 
                              target_names=label_encoder.classes_))

    # Per-class accuracy check
    print("\nPer-class Accuracy Check:")
    class_accuracy = classification_report(y_test, ensemble_classes, 
                                         target_names=label_encoder.classes_, 
                                         output_dict=True)

    for emotion in label_encoder.classes_:
        if emotion in class_accuracy:
            acc = class_accuracy[emotion]['precision']
            status = '✓' if acc > 0.75 else '✗'
            print(f"{emotion}: {status} ({acc:.2%})")
    
    # 6. Save ensemble models and preprocessing objects
    print("\n6. Saving ensemble models and preprocessing objects...")
    
    # Save deep learning models
    for i, (name, model) in enumerate(models):
        if name in ['dnn', 'cnn']:
            model.save(f'{MODEL_PATH}/ensemble_{name}_model.h5')
        else:
            joblib.dump(model, f'{MODEL_PATH}/ensemble_{name}_model.pkl')
    
    # Save preprocessing objects
    joblib.dump(scaler, f'{MODEL_PATH}/ensemble_scaler.pkl')
    joblib.dump(label_encoder, f'{MODEL_PATH}/ensemble_label_encoder.pkl')
    
    # Save model names for ensemble prediction
    model_names = [name for name, _ in models]
    joblib.dump(model_names, f'{MODEL_PATH}/ensemble_model_names.pkl')

    print("Ensemble models and preprocessing objects saved successfully!")
    
    # 7. Create confusion matrix visualization
    print("\n7. Creating confusion matrix...")
    cm = confusion_matrix(y_test, ensemble_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix - Ensemble Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{MODEL_PATH}/confusion_matrix_ensemble.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== Ensemble Pipeline completed successfully! ===")

if __name__ == "__main__":
    main() 