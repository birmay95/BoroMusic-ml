import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from tensorflow.keras.models import load_model

# === НАСТРОЙКИ ===
MODEL_PATH = 'baseline-model/model_baseline.keras'
DF_PATH = 'baseline-model/dataset_metadata.csv'
SCALER_PATH = 'baseline-model/scaler_baseline.joblib'

print("Загрузка модели и данных...")
model = load_model(MODEL_PATH, compile=False)
df = pd.read_csv(DF_PATH)
scaler = load(SCALER_PATH)

# === ВЫБОР ТРЕКА ===
# Берем список всех уникальных треков из метаданных
all_tracks = df['track_id'].unique()

# Давай для примера возьмем случайный трек из конца списка
# Можешь вписать сюда конкретный ID трека, например: TARGET_TRACK = 1234
TARGET_TRACK = 2058
print(f"Анализируем трек ID: {TARGET_TRACK}")

# Вытаскиваем все кусочки (окна) только для этого трека
track_df = df[df['track_id'] == TARGET_TRACK].copy()
# Сортируем по времени (т.к. у нас имена файлов вида trackID_15000.npy)
track_df['time_ms'] = track_df['filepath'].apply(lambda x: int(x.split('_')[-1].split('.')[0]))
track_df = track_df.sort_values('time_ms').reset_index(drop=True)

# === ПОДГОТОВКА ДАННЫХ ===
X_track = []
for filepath in track_df['filepath']:
    spec = np.load(filepath)
    X_track.append(spec)
X_track = np.array(X_track)

y_true = track_df[['valence', 'arousal']].values

# === ПРЕДСКАЗАНИЕ ===
print("Нейросеть слушает трек...")
y_pred_norm = model.predict(X_track)
y_pred = scaler.inverse_transform(y_pred_norm)

# Ось времени в секундах
time_axis = track_df['time_ms'] / 1000.0

# === ГРАФИК 1: ВРЕМЕННОЙ РЯД (Эмоции во времени) ===
plt.figure(figsize=(12, 6))

# График Arousal (Возбуждение)
plt.subplot(2, 1, 1)
plt.plot(time_axis, y_true[:, 1], label='Истинное Возбуждение (True)', color='gray', linestyle='dashed', linewidth=2)
plt.plot(time_axis, y_pred[:, 1], label='Предсказание нейросети (Pred)', color='purple', linewidth=2, marker='o', markersize=4)
plt.title(f'Динамика Возбуждения (Arousal) для трека {TARGET_TRACK}')
plt.ylabel('Возбуждение')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# График Valence (Валентность)
plt.subplot(2, 1, 2)
plt.plot(time_axis, y_true[:, 0], label='Истинная Валентность (True)', color='gray', linestyle='dashed', linewidth=2)
plt.plot(time_axis, y_pred[:, 0], label='Предсказание нейросети (Pred)', color='green', linewidth=2, marker='o', markersize=4)
plt.title(f'Динамика Валентности (Valence) для трека {TARGET_TRACK}')
plt.xlabel('Время (секунды)')
plt.ylabel('Валентность')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('track_timeseries.png', dpi=300)
plt.show()

# === ГРАФИК 2: ТРАЕКТОРИЯ ЭМОЦИЙ ===
plt.figure(figsize=(8, 8))

# Рисуем путь (соединяем точки линией)
plt.plot(y_pred[:, 0], y_pred[:, 1], color='blue', alpha=0.3, linewidth=1)
# Рисуем сами точки (чем позже точка, тем она крупнее/ярче)
scatter = plt.scatter(y_pred[:, 0], y_pred[:, 1], c=time_axis, cmap='viridis', s=50, edgecolors='black', label='Путь трека')

# Отмечаем Начало и Конец
plt.scatter(y_pred[0, 0], y_pred[0, 1], color='green', marker='s', s=100, label='Начало (15 сек)', zorder=5)
plt.scatter(y_pred[-1, 0], y_pred[-1, 1], color='red', marker='X', s=100, label='Конец (45 сек)', zorder=5)

# Добавляем Colorbar (шкала времени)
cbar = plt.colorbar(scatter)
cbar.set_label('Время (секунды)')

# Центральные оси
v_center = np.mean(y_true[:, 0])
a_center = np.mean(y_true[:, 1])
plt.axhline(a_center, color='red', linestyle='--', alpha=0.5)
plt.axvline(v_center, color='red', linestyle='--', alpha=0.5)

plt.title(f'Эмоциональная траектория трека {TARGET_TRACK}')
plt.xlabel('Valence (Валентность)')
plt.ylabel('Arousal (Возбуждение)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig('track_trajectory.png', dpi=300)
plt.show()

print("Готово! Графики конкретного трека сохранены.")