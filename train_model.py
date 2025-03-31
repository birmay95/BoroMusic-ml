import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from joblib import dump
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, LeakyReLU
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision

SR = 44100           # Частота дискретизации
SEGMENT_DURATION = 5 # Длительность сегмента в секундах
N_MELS = 64          # Количество мел-полос
N_FFT = 2048         # Размер окна FFT
HOP_LENGTH = 512     # Шаг окна
TIME_STEPS = 427     # Временные шаги для 5-секундных сегментов

# Загрузка метаданных
metadata = pd.read_csv('dataset1.csv')
metadata = metadata[['track_id', 'valence_mean', 'arousal_mean']]
metadata['track_id'] = metadata['track_id'].astype(int).astype(str)

# Нормализация меток
scaler = MinMaxScaler(feature_range=(0, 1))
metadata[['valence_mean', 'arousal_mean']] = scaler.fit_transform(
    metadata[['valence_mean', 'arousal_mean']]
)
dump(scaler, "scaler.joblib")

def process_audio(file_path):
    y, sr = librosa.load(file_path, sr=SR)
    
    # Сегментация на 5-секундные блоки
    segment_samples = SEGMENT_DURATION * SR
    segments = [
        y[i:i+segment_samples] 
        for i in range(0, len(y), segment_samples)
        if len(y[i:i+segment_samples]) == segment_samples
    ]
    
    # Генерация спектрограмм
    mel_specs = []
    for seg in segments:
        S = librosa.feature.melspectrogram(
            y=seg, sr=SR, n_mels=N_MELS, 
            n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        S_DB = librosa.power_to_db(S, ref=np.max)
        mel_specs.append(S_DB[:, :TIME_STEPS])
        
    return np.array(mel_specs)

# Обработка всех файлов
X, y = [], []
for idx, row in metadata.iterrows():
    file_path = os.path.join('DEAM_audio/MEMD_audio', f"{row['track_id']}.mp3")
    if os.path.exists(file_path):
        specs = process_audio(file_path)
        for spec in specs:
            X.append(spec)
            y.append([row['valence_mean'], row['arousal_mean']])
            
X = np.array(X)[..., np.newaxis]  # Добавление оси канала
y = np.array(y)

# Стратифицированное разделение по трекам
track_ids = metadata['track_id'].unique()
train_ids, temp_ids = train_test_split(track_ids, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

# Фильтрация данных
X_train = X[[i for i, t in enumerate(metadata['track_id']) if t in train_ids]]
X_val = X[[i for i, t in enumerate(metadata['track_id']) if t in val_ids]]
X_test = X[[i for i, t in enumerate(metadata['track_id']) if t in test_ids]]

y_train = y[[i for i, t in enumerate(metadata['track_id']) if t in train_ids]]
y_val = y[[i for i, t in enumerate(metadata['track_id']) if t in val_ids]]
y_test = y[[i for i, t in enumerate(metadata['track_id']) if t in test_ids]]

model = Sequential([
    Conv2D(32, (3,3), activation=None, input_shape=(N_MELS, TIME_STEPS, 1)),
    BatchNormalization(),
    LeakyReLU(),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation=None),
    BatchNormalization(),
    LeakyReLU(),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation=None),
    BatchNormalization(),
    LeakyReLU(),
    MaxPooling2D((2,2)),

    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    
    Dense(2, activation='linear')
])

print("Используемые устройства:")
print(tf.config.list_logical_devices())

optimizer = mixed_precision.LossScaleOptimizer(
    AdamW(learning_rate=1e-4, weight_decay=1e-4)
)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['mae']
)

class MemoryCleanup(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        tf.config.experimental.reset_memory_stats('GPU:0')

callbacks = [
    MemoryCleanup(),
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        min_delta=0.001,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        'model_full_time_steps.keras', 
        save_best_only=True,
        monitor='val_mae'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)

# Оценка на тестовом наборе
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Test MSE: {test_loss:.4f}, Test MAE: {test_mae:.4f}')

# Анализ распределения ошибок
errors = model.predict(X_test) - y_test
plt.figure(figsize=(10,6))
sns.histplot(errors[:,0], kde=True, label='Valence Errors')
sns.histplot(errors[:,1], kde=True, label='Arousal Errors')
plt.title('Error Distribution')
plt.legend()
plt.show()

# Визуализация истории обучения
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.legend()
plt.show()

# Предсказание для трека
def evaluate_track(track_id):
    try:
        print("Ищем track_id:", track_id)
        print("Все track_id в metadata:", metadata['track_id'].unique()[:10])

        idx = np.where(metadata['track_id'] == track_id)[0][0]
        print("Индекс найден:", idx)

        file_path = os.path.join('DEAM_audio/MEMD_audio', f"{track_id}.mp3")
        print("Путь к файлу:", file_path)
        print("Файл существует?", os.path.exists(file_path))

        spec = X[idx][np.newaxis, ...]
        pred = model.predict(spec)[0]
        true = y[idx]
        
        # Обратное преобразование меток
        pred_orig = scaler.inverse_transform([pred])[0]
        true_orig = scaler.inverse_transform([true])[0]
        
        # Визуализация
        plt.figure(figsize=(15, 5))
        
        # Спектрограмма
        plt.subplot(1, 2, 1)
        librosa.display.specshow(
            spec[0, ..., 0], 
            sr=SR, hop_length=HOP_LENGTH,
            x_axis='time', y_axis='mel'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel-Spectrogram (ID: {track_id})')
        
        # Valence-Arousal
        plt.subplot(1, 2, 2)
        plt.scatter(true_orig[0], true_orig[1], c='r', s=100, label='True')
        plt.scatter(pred_orig[0], pred_orig[1], c='b', s=100, label='Predicted')
        plt.xlim(1, 9)
        plt.ylim(1, 9)
        plt.grid()
        plt.legend()
        plt.title('Valence-Arousal Plane')
        
        plt.show()
        
        print(f'Track {track_id} Results:')
        print(f'True: Valence={true_orig[0]:.1f}, Arousal={true_orig[1]:.1f}')
        print(f'Pred: Valence={pred_orig[0]:.1f}, Arousal={pred_orig[1]:.1f}')
        
    except IndexError:
        print(f'Track {track_id} not found in dataset')

evaluate_track("47")