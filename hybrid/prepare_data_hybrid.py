import os
import math
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

SR = 22050
SEGMENT_DURATION = 5
STEP_SIZE = 0.5
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TIME_STEPS = math.ceil((SEGMENT_DURATION * SR) / HOP_LENGTH)

# Новая папка (можешь указать диск Windows, как делал раньше)
SAVE_DIR = "processed_spectrograms_hybrid" 
os.makedirs(SAVE_DIR, exist_ok=True)

arousal_df = pd.read_csv('DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/arousal.csv')
valence_df = pd.read_csv('DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/valence.csv')
arousal_dict = arousal_df.set_index('song_id').to_dict('index')
valence_dict = valence_df.set_index('song_id').to_dict('index')

metadata_list = []
track_list = list(arousal_dict.keys())

print(f"Извлечение гибридных данных (Спектрограмма + Математика) для {len(track_list)} треков...")
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
                    
                    # === 1. ГЕНЕРИРУЕМ СПЕКТРОГРАММУ (ГЛАЗА) ===
                    S = librosa.feature.melspectrogram(y=segment, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
                    mel_db = librosa.power_to_db(S, ref=np.max)
                    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
                    
                    if mel_norm.shape[1] >= TIME_STEPS:
                        mel_norm = mel_norm[:, :TIME_STEPS]
                        
                        file_name = f"{track_id}_{ms_start}.npy"
                        save_path = os.path.join(SAVE_DIR, file_name)
                        np.save(save_path, mel_norm)
                        
                        # === 2. ГЕНЕРИРУЕМ ТОЧНЫЕ ЦИФРЫ (КАЛЬКУЛЯТОР) ===
                        # 1. Темп (Удары в минуту)
                        tempo, _ = librosa.beat.beat_track(y=segment, sr=SR)
                        tempo = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
                        # 2. Громкость (RMS)
                        rms = np.mean(librosa.feature.rms(y=segment))
                        # 3. Спектральный Центроид (Яркость звука)
                        centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=SR))
                        # 4. Zero Crossing Rate (Количество "шума/перкуссии")
                        zcr = np.mean(librosa.feature.zero_crossing_rate(y=segment))
                        # 5. Спектральный Rolloff (Как высоко забираются частоты)
                        rolloff = np.mean(librosa.feature.spectral_rolloff(y=segment, sr=SR))
                        
                        # Записываем всё в наш "рецепт"
                        metadata_list.append({
                            'filepath': save_path,
                            'track_id': track_id,
                            'valence': mean_v,
                            'arousal': mean_a,
                            'tempo': tempo,
                            'rms': rms,
                            'centroid': centroid,
                            'zcr': zcr,
                            'rolloff': rolloff
                        })
            
            current_start += STEP_SIZE
            
    except Exception as e:
        pass

metadata_df = pd.DataFrame(metadata_list)
metadata_df.to_csv('dataset_metadata_hybrid.csv', index=False)
print("Готово! Извлечены спектрограммы и 5 математических параметров.")