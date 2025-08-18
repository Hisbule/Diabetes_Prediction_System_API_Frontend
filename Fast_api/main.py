from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Fast_api.schemas import diabetes
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("best_diabetes_model.joblib")
scaler = joblib.load("scaler.joblib")

app = FastAPI()

origins = [
    "http://localhost:5173",                   # local frontend (Vite dev)
    "https://diabetify-api.onrender.com"       # deployed frontend
]
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_cheak():
    return ({"status": "OK", "message": "API is running smoothly!"})
@app.get("/info")
async def get_info():
    return {
        "model type": "Random Forest Classifier",
        "model path": "best_diabetes_model.joblib",
        "scaler path": "scaler.joblib",
        "description": "This API predicts diabetes based on user input features.",
        "description": "API for predicting diabetes based on user input features.",
        "features": [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    }

@app.post("/predict")
async def predict_diabetes(data: diabetes):
  
    
    # Convert input to numpy array
    features = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, 
                          data.SkinThickness, data.Insulin, data.BMI,
                          data.DiabetesPedigreeFunction, data.Age]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    
    return {"Diabetes": bool(prediction)}
