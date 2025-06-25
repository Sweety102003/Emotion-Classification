import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = 'data'
MODEL_PATH = 'models'
os.makedirs(MODEL_PATH, exist_ok=True)

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
    parts = filename.replace('.wav', '').split('-')
    if len(parts) >= 7:
        emotion_code = parts[2]
        return emotion_map.get(emotion_code, 'unknown')
    return None

def extract_mel_spectrogram(file_path, n_mels=128, max_len=128):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # Pad or truncate to max_len
        if mel_db.shape[1] < max_len:
            pad_width = max_len - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0,0),(0,pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :max_len]
        return mel_db
    except Exception as e:
        print(f"[ERROR] Mel extraction failed for {file_path}: {e}")
        return None

def prepare_mel_dataset(df, n_mels=128, max_len=128):
    X = []
    valid_indices = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        mel = extract_mel_spectrogram(row['filepath'], n_mels, max_len)
        if mel is not None:
            X.append(mel)
            valid_indices.append(idx)
    X = np.array(X)
    X = X[..., np.newaxis]  # (num_samples, n_mels, max_len, 1)
    df_filtered = df.iloc[valid_indices].reset_index(drop=True)
    return X, df_filtered

def create_cnn_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    print("=== CNN Mel-spectrogram Emotion Classification Pipeline ===\n")
    # 1. Load and parse data
    audio_files = []
    for actor_dir in os.listdir(DATA_PATH):
        if actor_dir.startswith('Actor_'):
            actor_path = os.path.join(DATA_PATH, actor_dir)
            if os.path.isdir(actor_path):
                for file in os.listdir(actor_path):
                    if file.endswith('.wav'):
                        emotion = parse_filename(file)
                        if emotion and emotion != 'unknown':
                            audio_files.append({'filepath': os.path.join(actor_path, file), 'emotion': emotion})
    df = pd.DataFrame(audio_files)
    print(f"Total audio files: {len(df)}")
    print(df['emotion'].value_counts())

    # 2. Extract Mel-spectrograms
    print("\nExtracting Mel-spectrograms...")
    n_mels = 64
    max_len = 64
    X, df_filtered = prepare_mel_dataset(df, n_mels, max_len)
    print(f"Mel-spectrogram array shape: {X.shape}")

    # 3. Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_filtered['emotion'])
    print("Label classes:", label_encoder.classes_)

    # 4. Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # 5. Model
    input_shape = (n_mels, max_len, 1)
    num_classes = len(label_encoder.classes_)
    model = create_cnn_model(input_shape, num_classes)
    model.summary()

    # 6. Training
    EPOCHS = 50
    BATCH_SIZE = 16
    PATIENCE = 15
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=f'{MODEL_PATH}/cnn_mel_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # 7. Evaluation
    print("\nEvaluating on test set...")
    best_model = tf.keras.models.load_model(f'{MODEL_PATH}/cnn_mel_model.h5')
    test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    y_pred = best_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    f1_macro = f1_score(y_test, y_pred_classes, average='macro')
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

    # 8. Save label encoder
    joblib.dump(label_encoder, f'{MODEL_PATH}/cnn_mel_label_encoder.pkl')
    print("Saved model and label encoder.")

    # 9. Confusion matrix
    cm = tf.math.confusion_matrix(y_test, y_pred_classes).numpy()
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix - CNN Mel Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{MODEL_PATH}/confusion_matrix_cnn_mel.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main() 