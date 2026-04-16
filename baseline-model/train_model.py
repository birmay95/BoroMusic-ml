import os
import math
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from joblib import dump
from tensorflow.keras.utils import Sequence
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, LeakyReLU
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# === ОБНОВЛЕННЫЕ КОНСТАНТЫ ===
SR = 22050              # Пониженная частота (быстрее, меньше шума)
SEGMENT_DURATION = 5    # Длина куска (сек)
STEP_SIZE = 0.5         # Шаг скользящего окна (сек)
N_MELS = 128            # Высокое разрешение по частотам
N_FFT = 2048            
HOP_LENGTH = 512        
# Вычисляем ширину картинки автоматически: (5 сек * 22050 Гц) / 512
TIME_STEPS = math.ceil((SEGMENT_DURATION * SR) / HOP_LENGTH) 
BATCH_SIZE = 32

df = pd.read_csv('dataset_metadata.csv')

# Нормализация меток от 0 до 1
scaler = MinMaxScaler(feature_range=(0, 1))
df[['valence', 'arousal']] = scaler.fit_transform(df[['valence', 'arousal']])
dump(scaler, "scaler_baseline.joblib")

# Правильное разбиение по трекам (чтобы куски одного трека не попали в train и val одновременно)
unique_tracks = df['track_id'].unique()
train_ids, temp_ids = train_test_split(unique_tracks, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

# Разбиваем наш рецепт (DataFrame) на три части
train_df = df[df['track_id'].isin(train_ids)].reset_index(drop=True)
val_df = df[df['track_id'].isin(val_ids)].reset_index(drop=True)
test_df = df[df['track_id'].isin(test_ids)].reset_index(drop=True)

# === СОЗДАЕМ КЛАСС ГЕНЕРАТОРА (Data Generator) ===
class SpectrogramGenerator(Sequence):
    def __init__(self, dataframe, batch_size=32, shuffle=True):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indices)

    # Метод сообщает Keras, сколько батчей в одной эпохе
    def __len__(self):
        return math.ceil(len(self.dataframe) / self.batch_size)

    # Этот метод Keras вызывает каждый шаг, чтобы получить новые 32 картинки
    def __getitem__(self, idx):
        # Берем индексы для текущего батча (по 32 штуки)
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_data = self.dataframe.iloc[batch_indices]
        
        X, y = [], []
        for _, row in batch_data.iterrows():
            # Загружаем спектрограмму прямо с жесткого диска (работает за миллисекунды)
            spec = np.load(row['filepath'])
            X.append(spec)
            y.append([row['valence'], row['arousal']])
            
        X = np.array(X)[..., np.newaxis] # Добавляем канал
        y = np.array(y)
        return X, y

    # После каждой эпохи перемешиваем данные, чтобы сеть не привыкала к порядку
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# Создаем экземпляры генераторов
train_generator = SpectrogramGenerator(train_df, batch_size=BATCH_SIZE, shuffle=True)
val_generator = SpectrogramGenerator(val_df, batch_size=BATCH_SIZE, shuffle=False)
test_generator = SpectrogramGenerator(test_df, batch_size=BATCH_SIZE, shuffle=False)

# === 3. АРХИТЕКТУРА VGG-STYLE CNN ===
inputs = Input(shape=(N_MELS, TIME_STEPS, 1))

# Блок 1 (2 свертки -> Пулинг)
x = Conv2D(32, (3,3), padding='same', activation=None)(inputs)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2D(32, (3,3), padding='same', activation=None)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.2)(x) # Защита от переобучения

# Блок 2
x = Conv2D(64, (3,3), padding='same', activation=None)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2D(64, (3,3), padding='same', activation=None)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.2)(x)

# Блок 3
x = Conv2D(128, (3,3), padding='same', activation=None)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2D(128, (3,3), padding='same', activation=None)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.2)(x)

# Финальный "Мозг" (Эмбеддинги и Выход)
features = GlobalAveragePooling2D()(x)
features = Dense(128, activation='relu')(features)
features = Dropout(0.5)(features)

outputs = Dense(2, activation='linear')(features)

model = Model(inputs=inputs, outputs=outputs)

# === 4. КОМПИЛЯЦИЯ И ОБУЧЕНИЕ ===
optimizer = AdamW(learning_rate=1e-4, weight_decay=1e-4)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('model_baseline.keras', save_best_only=True, monitor='val_mae'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

print(f"Обучение на {len(train_df)} сэмплах...")

print("Начинаем обучение Baseline-модели...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=callbacks
)

# === 5. ОЦЕНКА И ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ===
print("Оценка модели на тестовой выборке...")
test_loss, test_mae = model.evaluate(test_generator, verbose=0)
print(f'Test MSE (Loss): {test_loss:.4f}')
print(f'Test MAE: {test_mae:.4f}')

# Получаем предсказания
y_pred = model.predict(test_generator)

# Возвращаем значения из нормализованных (0-1) обратно в оригинальную шкалу датасета
y_test_orig = scaler.inverse_transform(test_df[['valence', 'arousal']].values)
y_pred_orig = scaler.inverse_transform(y_pred)

# --- ГРАФИК 1: История обучения (Loss и MAE) ---
plt.figure(figsize=(14, 5))

# График MSE (Loss)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Обучение (Train Loss)', color='blue')
plt.plot(history.history['val_loss'], label='Валидация (Val Loss)', color='orange')
plt.title('График функции потерь (MSE)')
plt.xlabel('Эпохи')
plt.ylabel('Ошибка (MSE)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# График MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Обучение (Train MAE)', color='blue')
plt.plot(history.history['val_mae'], label='Валидация (Val MAE)', color='orange')
plt.title('График абсолютной ошибки (MAE)')
plt.xlabel('Эпохи')
plt.ylabel('Ошибка (MAE)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=300)
plt.show()

# --- ГРАФИК 2: Распределение ошибок (Гистограмма) ---
# Считаем разницу между предсказанием и реальностью
errors_valence = y_pred_orig[:, 0] - y_test_orig[:, 0]
errors_arousal = y_pred_orig[:, 1] - y_test_orig[:, 1]

plt.figure(figsize=(10, 5))
sns.histplot(errors_valence, color="green", kde=True, label="Ошибки Валентности (Valence)", alpha=0.5)
sns.histplot(errors_arousal, color="purple", kde=True, label="Ошибки Возбуждения (Arousal)", alpha=0.5)
plt.axvline(0, color='red', linestyle='dashed', linewidth=2) # Линия идеального совпадения
plt.title('Распределение ошибок предсказания')
plt.xlabel('Отклонение от истинного значения')
plt.ylabel('Количество сегментов')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('error_distribution.png', dpi=300)
plt.show()

# --- ГРАФИК 3: Пространство Valence-Arousal (Диаграмма рассеяния) ---
plt.figure(figsize=(8, 8))

# Рисуем реальные точки (серые) и предсказанные (синие)
plt.scatter(y_test_orig[:, 0], y_test_orig[:, 1], c='gray', alpha=0.5, label='Истинные значения', marker='o')
plt.scatter(y_pred_orig[:, 0], y_pred_orig[:, 1], c='blue', alpha=0.5, label='Предсказания модели', marker='x')

# Добавляем центральные оси (перекрестие)
v_center = np.mean(y_test_orig[:, 0]) # Центр облака точек по Valence
a_center = np.mean(y_test_orig[:, 1]) # Центр облака точек по Arousal
plt.axhline(a_center, color='red', linestyle='--')
plt.axvline(v_center, color='red', linestyle='--')

plt.title('Распределение предсказаний в пространстве Valence-Arousal')
plt.xlabel('Valence (Валентность)')
plt.ylabel('Arousal (Возбуждение)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig('valence_arousal_plane.png', dpi=300)
plt.show()