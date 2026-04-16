import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from joblib import load  # ВАЖНО: используем load вместо dump
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import Huber

BATCH_SIZE = 32

# === 1. ВОССТАНАВЛИВАЕМ ДАННЫЕ (ТОЛЬКО TEST) ===
print("Загрузка метаданных и скейлера...")
df = pd.read_csv('dataset_metadata_dyn.csv')

# ЗАГРУЖАЕМ сохраненный скейлер, чтобы не сбить нормализацию!
scaler = load("scaler_rgb.joblib")

# Точно так же разбиваем треки, чтобы получить идентичный test_df
unique_tracks = df['track_id'].unique()
train_ids, temp_ids = train_test_split(unique_tracks, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

test_df = df[df['track_id'].isin(test_ids)].reset_index(drop=True)

# === 2. КЛАСС ГЕНЕРАТОРА (Копируем из прошлого кода) ===
class SpectrogramDynGenerator(Sequence):
    def __init__(self, dataframe, batch_size=32, shuffle=True):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self.dataframe) / self.batch_size)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_data = self.dataframe.iloc[batch_indices]
        
        X, y, weights = [], [], []
        for _, row in batch_data.iterrows():
            spec = np.load(row['filepath'])
            v = row['valence']
            a = row['arousal']
            X.append(spec)
            y.append([v, a])
            
            distance = math.sqrt((v - 0.5)**2 + (a - 0.5)**2)
            weight = 1.0 + (distance * 5.0)
            weights.append(weight)
            
        return np.array(X), np.array(y), np.array(weights)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# Создаем ТОЛЬКО тестовый генератор (без shuffle, чтобы предсказания совпадали по порядку)
test_generator = SpectrogramDynGenerator(test_df, batch_size=BATCH_SIZE, shuffle=False)

# === 3. ЗАГРУЗКА ГОТОВОЙ МОДЕЛИ ===
print("Загрузка обученной модели с диска...")
# custom_objects нужен, чтобы Keras понял, что мы использовали Huber Loss
model = load_model('model_rgb.keras', compile=False)
model.compile(optimizer='adam', loss=Huber(delta=0.1), metrics=['mae', 'mse'])

# === 4. ОЦЕНКА И ПОСТРОЕНИЕ ГРАФИКОВ ===
print("Оценка модели на тестовой выборке...")
# ИСПРАВЛЕННАЯ СТРОЧКА: распаковываем 3 значения (Loss, MAE, MSE)
test_loss, test_mae, test_mse = model.evaluate(test_generator, verbose=1)
print(f'\nTest Huber Loss: {test_loss:.4f}')
print(f'Test MAE: {test_mae:.4f}')
print(f'Test MSE: {test_mse:.4f}')

print("Выполнение предсказаний...")
y_pred = model.predict(test_generator)
y_test_orig = scaler.inverse_transform(test_df[['valence', 'arousal']].values)
y_pred_orig = scaler.inverse_transform(y_pred)

# --- ГРАФИК 1: Распределение ошибок ---
errors_valence = y_pred_orig[:, 0] - y_test_orig[:, 0]
errors_arousal = y_pred_orig[:, 1] - y_test_orig[:, 1]

plt.figure(figsize=(10, 5))
sns.histplot(errors_valence, color="green", kde=True, label="Ошибки Валентности (Valence)", alpha=0.5)
sns.histplot(errors_arousal, color="purple", kde=True, label="Ошибки Возбуждения (Arousal)", alpha=0.5)
plt.axvline(0, color='red', linestyle='dashed', linewidth=2)
plt.title('Распределение ошибок предсказания')
plt.xlabel('Отклонение от истинного значения')
plt.ylabel('Количество сегментов')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('error_distribution_advanced.png', dpi=300)
plt.show()

# --- ГРАФИК 2: Пространство Valence-Arousal ---
plt.figure(figsize=(8, 8))
plt.scatter(y_test_orig[:, 0], y_test_orig[:, 1], c='gray', alpha=0.5, label='Истинные значения', marker='o')
plt.scatter(y_pred_orig[:, 0], y_pred_orig[:, 1], c='blue', alpha=0.5, label='Предсказания модели', marker='x')

v_center = np.mean(y_test_orig[:, 0])
a_center = np.mean(y_test_orig[:, 1])
plt.axhline(a_center, color='red', linestyle='--')
plt.axvline(v_center, color='red', linestyle='--')

plt.title('Распределение предсказаний (С учетом весов и Huber Loss)')
plt.xlabel('Valence (Валентность)')
plt.ylabel('Arousal (Возбуждение)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig('valence_arousal_plane_advanced.png', dpi=300)
plt.show()

print("Готово! Графики сохранены.")