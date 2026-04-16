import os
import math
import numpy as np
import pandas as pd
import librosa
import cv2  # НОВОЕ: Для изменения размера Хромаграммы
from tqdm import tqdm

SR = 22050
SEGMENT_DURATION = 5
STEP_SIZE = 0.5
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TIME_STEPS = math.ceil((SEGMENT_DURATION * SR) / HOP_LENGTH)

# НОВАЯ ПАПКА для 3-канальных данных
SAVE_DIR = "/run/media/mishail/Новый том/STUDY/diploma/ml_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

print("Загрузка динамических меток...")
arousal_df = pd.read_csv('DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/arousal.csv')
valence_df = pd.read_csv('DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/valence.csv')
arousal_dict = arousal_df.set_index('song_id').to_dict('index')
valence_dict = valence_df.set_index('song_id').to_dict('index')

metadata_list = []
track_list = list(arousal_dict.keys())

print(f"Начинаем извлечение Акустического RGB для {len(track_list)} треков...")
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
                    
                    # === 🔴 КАНАЛ 1: Mel-Spectrogram ===
                    S = librosa.feature.melspectrogram(y=segment, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
                    mel_db = librosa.power_to_db(S, ref=np.max)
                    # Нормализация 0-1
                    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
                    
                    # === 🔵 КАНАЛ 2: MFCC (Тембр) ===
                    # Указываем n_mfcc = N_MELS (128), чтобы высота матрицы сразу совпала
                    mfcc = librosa.feature.mfcc(y=segment, sr=SR, n_mfcc=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
                    mfcc_norm = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min() + 1e-8)
                    
                    # === 🟢 КАНАЛ 3: Chromagram (Ноты и Гармония) ===
                    chroma = librosa.feature.chroma_stft(y=segment, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
                    # Хрома имеет высоту 12. Растягиваем её до 128 (N_MELS) с помощью OpenCV
                    chroma_resized = cv2.resize(chroma, (chroma.shape[1], N_MELS), interpolation=cv2.INTER_NEAREST)
                    chroma_norm = (chroma_resized - chroma_resized.min()) / (chroma_resized.max() - chroma_resized.min() + 1e-8)
                    
                    # Обрезаем лишние кадры по ширине (если librosa отдала больше)
                    if mel_norm.shape[1] >= TIME_STEPS:
                        mel_norm = mel_norm[:, :TIME_STEPS]
                        mfcc_norm = mfcc_norm[:, :TIME_STEPS]
                        chroma_norm = chroma_norm[:, :TIME_STEPS]
                        
                        # === СБОРКА АКУСТИЧЕСКОГО RGB ===
                        # Склеиваем 3 канала. Получаем матрицу (128, 216, 3)
                        acoustic_rgb = np.stack([mel_norm, chroma_norm, mfcc_norm], axis=-1)
                        
                        file_name = f"{track_id}_{ms_start}.npy"
                        save_path = os.path.join(SAVE_DIR, file_name)
                        np.save(save_path, acoustic_rgb)
                        
                        metadata_list.append({
                            'filepath': save_path,
                            'track_id': track_id,
                            'valence': mean_v,
                            'arousal': mean_a
                        })
            
            current_start += STEP_SIZE
            
    except Exception as e:
        pass

metadata_df = pd.DataFrame(metadata_list)
metadata_df.to_csv('dataset_metadata_rgb.csv', index=False)
print(f"Готово! Извлечено и сохранено {len(metadata_df)} 3-канальных спектрограмм.")