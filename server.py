import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from joblib import load
from cachetools import TTLCache
from io import BytesIO

app = FastAPI()

DATASET_PATH = "music_db.csv"
MODEL_PATH = "model.keras"

SR = 44100           
SEGMENT_DURATION = 5 
N_MELS = 64         
N_FFT = 2048        
HOP_LENGTH = 512    
TIME_STEPS = 457    

ALLOWED_CONTENT_TYPES = ["audio/mpeg", "audio/wav"]
MAX_FILE_SIZE_MB = 50

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("The model file was not found!")
    
model = tf.keras.models.load_model(MODEL_PATH)

scaler_initialized = False
try:
    scaler = load("scaler.joblib")  
    scaler_initialized = True
except FileNotFoundError:
    exit("scaler.joblib not found!")

def extract_features(y: np.ndarray):

    segment_samples = SEGMENT_DURATION * SR
    segments = [
        y[i:i+segment_samples] 
        for i in range(0, len(y), segment_samples)
        if len(y[i:i+segment_samples]) == segment_samples
    ]
    
    mel_specs = []
    for seg in segments:
        S = librosa.feature.melspectrogram(
            y=seg, sr=SR, n_mels=N_MELS, 
            n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        S_DB = librosa.power_to_db(S, ref=np.max)
        mel_specs.append(S_DB[:, :TIME_STEPS])
        
    return np.array(mel_specs)[..., np.newaxis]

def predict_valence_arousal(y: np.ndarray):
    
    features = extract_features(y)

    if len(features) == 0:
        raise ValueError("Error: couldn't extract the signs")

    predictions = model.predict(features)
    valence, arousal = np.mean(predictions, axis=0)

    transformed_values = scaler.inverse_transform([[valence, arousal]])[0]
    print(f"Reverse conversion: {transformed_values}")
    valence, arousal = transformed_values

    return valence, arousal

recommendation_cache = TTLCache(maxsize=1000, ttl=300)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MusicEmotionAPI")

@app.post("/upload_track")
async def upload_track(file: UploadFile = File(...), track_id: str = Form(...)):

    logger.info(f"File received {file.filename} with track_id={track_id}")

    if not track_id:
        raise HTTPException(status_code=400, detail="track_id is required")
    if not track_id.isdigit():
        raise HTTPException(400, "track_id must be a number.")
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(400, "Only MP3/WAV files are supported.")
    
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"Maximum file size is {MAX_FILE_SIZE_MB}MB")
    
    try:
        audio_bytes = await file.read()
        buffer = BytesIO(audio_bytes)

        y, sr = librosa.load(buffer, sr=SR)

        valence, arousal = predict_valence_arousal(y)

    except librosa.LibrosaError as e:
        logger.error(f"Audio processing error: {str(e)}")
        raise HTTPException(415, "Invalid audio file")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    finally:
        buffer.close()
        file.file.close()

    df = pd.read_csv(DATASET_PATH)
    df.loc[len(df)] = {"track_id": track_id, "valence": valence, "arousal": arousal}
    df.to_csv(DATASET_PATH, index=False)

    return {"message": "Track added", "valence": float(valence), "arousal": float(arousal)}

@app.post("/recommend")
async def recommend(track_id: str = Form(...)):

    logger.info(f"Received a recommendation request for the track: {track_id}")

    df = pd.read_csv(DATASET_PATH)

    if track_id not in df["track_id"].astype(str).values:
        if track_id in recommendation_cache:
            del recommendation_cache[track_id]
            logger.warning(f"Removed orphaned track {track_id} from cache")
        raise HTTPException(status_code=400, detail="Track not found")
    
    if track_id in recommendation_cache:
        logger.info(f"Returning cached recommendations for {track_id}")
        return recommendation_cache[track_id]

    if track_id not in df["track_id"].astype(str).values:
        raise HTTPException(status_code=400, detail="Track not found")

    track = df[df["track_id"] == int(track_id)]

    if track.empty:
        raise HTTPException(status_code=400, detail="The track was not found in the database")

    track_vec = np.array([track.iloc[0]["valence"], track.iloc[0]["arousal"]])

    df["distance"] = df.apply(lambda row: np.linalg.norm(track_vec - np.array([row["valence"], row["arousal"]])), axis=1)
    recommendations = df.sort_values("distance").iloc[1:6]

    recommended_tracks = []
    for _, row in recommendations.iterrows():
        recommended_tracks.append({
            "track_id": row["track_id"],
            "valence": float(row["valence"]),
            "arousal": float(row["arousal"])
        })
    recommendation_cache[track_id] = {"recommendations": recommended_tracks}
    return {"recommendations": recommended_tracks}

class TrackDeleteRequest(BaseModel):
    track_id: str

@app.post("/delete_track")
async def delete_track(request: TrackDeleteRequest):

    track_id = request.track_id
    logger.info(f"Request to delete a track: {track_id}")
    
    try:
        if track_id in recommendation_cache:
            del recommendation_cache[track_id]
            logger.info(f"Track {track_id} removed from cache")

        df = pd.read_csv(DATASET_PATH)
        
        if track_id not in df["track_id"].astype(str).values:
            raise HTTPException(
                status_code=404,
                detail="The track was not found in the database"
            )
            
        initial_count = len(df)
        df = df[df["track_id"].astype(str) != track_id]
        
        if len(df) == initial_count:
            raise HTTPException(
                status_code=500,
                detail="Error when deleting a track"
            )
            
        df.to_csv(DATASET_PATH, index=False)
        logger.info(f"Track {track_id} successfully deleted")
        
        return {
            "message": "The track was deleted successfully",
            "deleted_track_id": track_id
        }
        
    except pd.errors.EmptyDataError:
        logger.error("The database is empty or corrupted")
        raise HTTPException(
            status_code=500,
            detail="Database reading error"
        )
        
    except Exception as e:
        logger.error(f"Error when deleting: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
