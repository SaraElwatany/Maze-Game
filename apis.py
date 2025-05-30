import io
import cv2
import base64
import joblib
import uvicorn
import numpy as np
import pandas as pd
from PIL import Image
import mediapipe as mp
from fastapi import Body
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware



# Create the FastAPI app instance
app = FastAPI()



# Load your ML model
model = joblib.load("model.pkl") 

# Feature columns (e.g., x0, y0, z0, ..., x20, y20, z20)
feature_names = [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']]

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)



# Allow frontend access 
app.add_middleware(
                    CORSMiddleware,
                    allow_origins=["*"],  # Allow from all sources 
                    allow_methods=["*"],
                    allow_headers=["*"],
                )


def center_landmarks(df, col):
    mean = df[col].mean()
    return df[col] - mean


def normalize_landmarks(df, col):
    max_val = df[col].abs().max()
    return df[col] / max_val if max_val != 0 else df[col]



@app.get("/")
def home():
    return {"message": "Welcome to Home"}


@app.get("/health")
def check_health():
    return {"status": "healthy"}



@app.post("/predict")
async def predict(request: Request):

    data = await request.json()
    img_b64 = data["image"]

    # Decode base64 image to OpenCV format
    img_bytes = base64.b64decode(img_b64.split(",")[1])
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Detect hand landmarks
    results = hands.process(image_np)
    if not results.multi_hand_landmarks:
        return {"label": None}

    # Extract landmarks
    landmarks = results.multi_hand_landmarks[0]
    coords = [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
    flat = np.array(coords).flatten()

    # Convert to DataFrame and preprocess
    df = pd.DataFrame([flat], columns=feature_names)
    df_centered = df.apply(lambda col: center_landmarks(df, col), axis=0)
    df_normalized = df_centered.apply(lambda col: normalize_landmarks(df_centered, col), axis=0)

    # Predict gesture
    prediction = model.predict(df_normalized)[0]

    return {"label": prediction}

