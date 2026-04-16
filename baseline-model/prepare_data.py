import os
import math
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm  # Библиотека для красивой полосы загрузки (pip install tqdm)

SR = 22050
SEGMENT_DURATION = 5
STEP_SIZE = 0.5
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TIME_STEPS = math.ceil((SEGMENT_DURATION * SR) / HOP_LENGTH)

# Создаем папку для готовых спектрограмм
SAVE_DIR = "processed_spectrograms"
os.makedirs(SAVE_DIR, exist_ok=True)

print("Загрузка динамических меток...")
arousal_df = pd.read_csv('DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/arousal.csv')
valence_df = pd.read_csv('DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/valence.csv')
arousal_dict = arousal_df.set_index('song_id').to_dict('index')
valence_dict = valence_df.set_index('song_id').to_dict('index')

# Здесь мы будем хранить наш "рецепт" для генератора
metadata_list = []

track_list = list(arousal_dict.keys())

print(f"Начинаем обработку {len(track_list)} треков...")
# tqdm добавит красивый прогресс-бар в консоли
for track_id in tqdm(track_list):
    file_path = os.path.join('DEAM_audio/MEMD_audio', f"{track_id}.mp3")
    if not os.path.exists(file_path): continue
        
    try:
        y_audio, sr = librosa.load(file_path, sr=SR)
        total_duration = len(y_audio) / sr
        
        current_start = 15.0
        while current_start + SEGMENT_DURATION <= total_duration:
            current_end = current_start + SEGMENT_DURATION
            
            ms_start = int(current_start * 1000)
            ms_end = int(current_end * 1000)
            cols = [f"sample_{ms}ms" for ms in range(ms_start, ms_end, 500)]
            
            if all(col in arousal_dict[track_id] for col in cols):
                a_vals = [arousal_dict[track_id][col] for col in cols]
                v_vals = [valence_dict[track_id][col] for col in cols]
                
                if not any(pd.isna(a_vals)) and not any(pd.isna(v_vals)):
                    mean_a = np.mean(a_vals)
                    mean_v = np.mean(v_vals)
                    
                    start_sample = int(current_start * SR)
                    end_sample = int(current_end * SR)
                    segment = y_audio[start_sample:end_sample]
                    
                    S = librosa.feature.melspectrogram(y=segment, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
                    S_DB = librosa.power_to_db(S, ref=np.max)
                    
                    if S_DB.shape[1] >= TIME_STEPS:
                        S_DB = S_DB[:, :TIME_STEPS]
                        
                        # СОХРАНЯЕМ СПЕКТРОГРАММУ НА ДИСК
                        file_name = f"{track_id}_{ms_start}.npy"
                        save_path = os.path.join(SAVE_DIR, file_name)
                        np.save(save_path, S_DB)
                        
                        # Записываем в "рецепт" путь к файлу и его метки
                        metadata_list.append({
                            'filepath': save_path,
                            'track_id': track_id,
                            'valence': mean_v,
                            'arousal': mean_a
                        })
            
            current_start += STEP_SIZE
            
    except Exception as e:
        pass # Игнорируем ошибки

# Сохраняем рецепт в CSV, чтобы потом быстро его загружать при обучении
metadata_df = pd.DataFrame(metadata_list)
metadata_df.to_csv('dataset_metadata.csv', index=False)
print(f"Готово! Извлечено и сохранено {len(metadata_df)} спектрограмм.")