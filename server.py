import os
import numpy as np
import librosa
import tensorflow as tf
import logging
import psycopg2
from pgvector.psycopg2 import register_vector
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from joblib import load
from io import BytesIO

app = FastAPI()

MODEL_PATH = "model.keras"

SR = 44100           
SEGMENT_DURATION = 5 
N_MELS = 64         
N_FFT = 2048        
HOP_LENGTH = 512    
TIME_STEPS = 457    

ALLOWED_CONTENT_TYPES = ["audio/mpeg", "audio/wav"]

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "music_platform")
DB_USER = os.getenv("DB_USER", "mishail")
DB_PASS = os.getenv("DB_PASS", "12348765")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("The model file was not found!")
    
model = tf.keras.models.load_model(MODEL_PATH)

try:
    scaler = load("scaler.joblib")  
except FileNotFoundError:
    exit("scaler.joblib not found!")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MusicEmotionAPI")

def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )
    register_vector(conn)
    return conn

def extract_features(y: np.ndarray):
    """Преобразует временной ряд в массив мел-спектрограмм"""
    segment_samples = SEGMENT_DURATION * SR
    segments = [
        y[i:i+segment_samples] 
        for i in range(0, len(y), segment_samples)
        if len(y[i:i+segment_samples]) == segment_samples
    ]
    
    mel_specs = []
    for seg in segments:
        S = librosa.feature.melspectrogram(
            y=seg, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        S_DB = librosa.power_to_db(S, ref=np.max)
        mel_specs.append(S_DB[:, :TIME_STEPS])
        
    return np.array(mel_specs)[..., np.newaxis]

def predict_valence_arousal(y: np.ndarray):
    """Вычисляет характеристики через модель и выполняет обратное масштабирование"""
    features = extract_features(y)
    if len(features) == 0:
        raise ValueError("Error: couldn't extract the features")

    predictions = model.predict(features)
    valence, arousal = np.mean(predictions, axis=0)

    transformed_values = scaler.inverse_transform([[valence, arousal]])[0]
    return transformed_values[0], transformed_values[1]


@app.post("/api/v1/upload_track")
async def upload_track(file: UploadFile = File(...), track_id: str = Form(...)):
    logger.info(f"UploadTrack: File received {file.filename} with track_id={track_id}")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(400, detail="UploadTrackException: Only MP3/WAV files are supported.")
    
    try:
        audio_bytes = await file.read()
        buffer = BytesIO(audio_bytes)
        y, sr = librosa.load(buffer, sr=SR)

        valence, arousal = predict_valence_arousal(y)
        mood_vector = np.array([valence, arousal])

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO feature_vectors (id, track_id, mood_vector)
                    VALUES (gen_random_uuid(), %s::uuid, %s)
                    ON CONFLICT (track_id) 
                    DO UPDATE SET mood_vector = EXCLUDED.mood_vector;
                """, (track_id, mood_vector))
            conn.commit()

        return {"message": "Track vector added/updated", "valence": float(valence), "arousal": float(arousal)}
    except Exception as e:
        logger.error(f"UploadTrackException: {str(e)}")
        raise HTTPException(status_code=500, detail=f"UploadTrackException: {str(e)}")

class RecommendRequest(BaseModel):
    track_id: str
    excluded_ids: list[str] = []

@app.post("/api/v1/recommend")
async def recommend(request: RecommendRequest):
    logger.info(f"Recommend: Request for track={request.track_id}, excluded={len(request.excluded_ids)}")

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT mood_vector FROM feature_vectors WHERE track_id = %s::uuid", (request.track_id,))
                res = cur.fetchone()
                
                if not res or res[0] is None:
                    logger.error(f"TrackFeatureNotFoundException: target track {request.track_id} not found")
                    raise HTTPException(404, detail="TrackFeatureNotFoundException: Vector not found")
                
                target_vector = res[0]

                query = """
                    SELECT track_id, mood_vector
                    FROM feature_vectors
                    WHERE track_id != %s::uuid AND mood_vector IS NOT NULL
                """
                params = [request.track_id]

                if request.excluded_ids:
                    query += " AND track_id != ALL(%s::uuid[]) "
                    params.append(request.excluded_ids)

                query += " ORDER BY mood_vector <-> %s LIMIT 5 "
                params.append(target_vector)

                cur.execute(query, tuple(params))

                recommendations = [
                    {"track_id": str(row[0]), "valence": float(row[1][0]), "arousal": float(row[1][1])}
                    for row in cur.fetchall()
                ]
        return {"recommendations": recommendations}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DB Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")

class PersonalRecommendRequest(BaseModel):
    liked_track_ids: list[str] = Field(default=[], max_length=50)
    excluded_ids: list[str] = []

@app.post("/api/v1/recommend/personal")
async def recommend_personal(request: PersonalRecommendRequest):
    logger.info(f"PersonalFeed: liked={len(request.liked_track_ids)}, excluded={len(request.excluded_ids)}")

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                
                centroid = None
                
                if not request.liked_track_ids:
                    logger.warning("UserPreferencesNotFoundException: liked list is empty. Fallback to center.")
                    centroid = np.array([5.0, 5.0])
                else:
                    cur.execute("""
                        SELECT mood_vector FROM feature_vectors 
                        WHERE track_id = ANY(%s::uuid[]) AND mood_vector IS NOT NULL
                    """, (request.liked_track_ids,))
                    
                    rows = cur.fetchall()
                    if rows:
                        vectors = [row[0] for row in rows]
                        
                        logger.info(f"Retrieved {len(vectors)} valid mood vectors from user history.")
                        logger.info("Calculating arithmetic mean of coordinates for Valence and Arousal axes...")
                        
                        centroid = np.mean(vectors, axis=0)
                        
                        logger.info(f"Taste centroid successfully generated: Valence={centroid[0]:.4f}, Arousal={centroid[1]:.4f}")
                        # -----------------------------------------------
                    else:
                        logger.warning("UserPreferencesNotFoundException: valid vectors not found. Fallback to center.")
                        centroid = np.array([5.0, 5.0])

                query = """
                    SELECT track_id, mood_vector
                    FROM feature_vectors
                    WHERE mood_vector IS NOT NULL
                """
                params = []

                if request.excluded_ids:
                    query += " AND track_id != ALL(%s::uuid[]) "
                    params.append(request.excluded_ids)

                query += " ORDER BY mood_vector <-> %s LIMIT 30 "
                params.append(centroid)

                cur.execute(query, tuple(params))

                recommendations = [
                    {"track_id": str(row[0]), "valence": float(row[1][0]), "arousal": float(row[1][1])}
                    for row in cur.fetchall()
                ]
        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"DB Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")

@app.delete("/api/v1/delete_track/{track_id}")
async def delete_track(track_id: str):
    logger.info(f"DeleteTrack: Request to delete features for track {track_id}")
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM feature_vectors WHERE track_id = %s::uuid", (track_id,))
            conn.commit()
        return {"message": f"The features for track {track_id} were deleted successfully"}
    except Exception as e:
        logger.error(f"Error when deleting: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during deletion")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)