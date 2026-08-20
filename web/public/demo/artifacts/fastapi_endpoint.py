import asyncio
import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter
from pydantic import BaseModel, Field

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
prediction_counter = Counter(
    'predictions_total',
    'Total number of predictions made',
    ['model_version']
)

# Global model storage
model = None
model_version = "1.0.0"


class PredictRequest(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str


class PredictResponse(BaseModel):
    prediction: str
    probability: float
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model
    try:
        model = joblib.load('model.pkl')
        logger.info(f"Model loaded successfully. Version: {model_version}")
    except FileNotFoundError:
        logger.error("model.pkl not found. Please ensure the model file exists.")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Application shutting down")


app = FastAPI(title="Churn Prediction API", version=model_version, lifespan=lifespan)


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": model_version
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Make a prediction using the loaded model
    """
    start_time = time.time()
    
    try:
        # Convert request to dictionary
        request_dict = request.model_dump()
        
        # Create input hash for logging
        request_json = json.dumps(request_dict, sort_keys=True)
        input_hash = hashlib.md5(request_json.encode()).hexdigest()
        
        # Prepare features for model
        # Convert TotalCharges to float, handling any non-numeric values
        try:
            total_charges = float(request.TotalCharges)
        except (ValueError, TypeError):
            total_charges = 0.0
        
        # Create feature vector in the order expected by the model
        features_dict = {
            'customerID': request.customerID,
            'gender': request.gender,
            'SeniorCitizen': request.SeniorCitizen,
            'Partner': request.Partner,
            'Dependents': request.Dependents,
            'tenure': request.tenure,
            'PhoneService': request.PhoneService,
            'MultipleLines': request.MultipleLines,
            'InternetService': request.InternetService,
            'OnlineSecurity': request.OnlineSecurity,
            'OnlineBackup': request.OnlineBackup,
            'DeviceProtection': request.DeviceProtection,
            'TechSupport': request.TechSupport,
            'StreamingTV': request.StreamingTV,
            'StreamingMovies': request.StreamingMovies,
            'Contract': request.Contract,
            'PaperlessBilling': request.PaperlessBilling,
            'PaymentMethod': request.PaymentMethod,
            'MonthlyCharges': request.MonthlyCharges,
            'TotalCharges': total_charges
        }
        
        # Create DataFrame for prediction
        df = pd.DataFrame([features_dict])
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Get prediction probability
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df)[0]
            # Get probability of the predicted class
            class_index = list(model.classes_).index(prediction)
            probability = float(probabilities[class_index])
        else:
            probability = 1.0
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Log prediction with structured format
        logger.info(
            f"Prediction made - Input Hash: {input_hash}, "
            f"Prediction: {prediction}, "
            f"Probability: {probability:.4f}, "
            f"Latency: {latency:.4f}s, "
            f"Timestamp: {datetime.utcnow().isoformat()}"
        )
        
        # Increment prediction counter
        prediction_counter.labels(model_version=model_version).inc()
        
        return PredictResponse(
            prediction=str(prediction),
            probability=probability,
            model_version=model_version
        )
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "fastapi_endpoint:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )