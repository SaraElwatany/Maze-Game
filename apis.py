import time
import logging
import joblib
import uvicorn
import statistics
import pandas as pd

from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

# Prometheus client imports
from prometheus_client import Summary, Counter, generate_latest, CONTENT_TYPE_LATEST


# Set up logging for monitoring to file
logging.basicConfig(
                      filename="metrics.log",
                      level=logging.INFO,
                      format="%(asctime)s - %(levelname)s - %(message)s"
                    )



# Prometheus metrics
inference_time_metric = Summary('model_inference_time_seconds', 'Time spent on model inference')
model_failures = Counter('model_failures_total', 'Number of model prediction failures')
data_input_errors = Counter('data_input_errors_total', 'Invalid data inputs received')
http_requests_total = Counter('http_requests_total', 'Total HTTP requests to endpoints', ['method', 'endpoint'])



# Create the FastAPI app instance
app = FastAPI()


# Load the ML models
rf_model = joblib.load("rf_model.pkl")
svm_poly_model = joblib.load("svm_poly_model.pkl")
svm_rbf_model = joblib.load("svm_rbf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
xgb_label_encoder = joblib.load("xgb_label_encoder.pkl")


# Define feature columns (e.g., x1, y1, z1, ..., x21, y21, z21)
feature_names = [f'{axis}{i}' for i in range(1, 22) for axis in ['x', 'y', 'z']]


# Allow CORS (all origins)
app.add_middleware(
                    CORSMiddleware,
                    allow_origins=["*"],
                    allow_methods=["*"],
                    allow_headers=["*"],
                  )




@app.middleware("http")
async def log_server_errors(request: Request, call_next):
    try:
        response = await call_next(request)
        if response.status_code >= 500:
            logging.error(f"5xx Error - Path: {request.url.path}, Status Code: {response.status_code}")
        return response
    except Exception as e:
        logging.error(f"Unhandled Exception - Path: {request.url.path}, Error: {str(e)}")
        raise e



def center_landmarks(col, landmarks_df):
    col_name = col.name
    if 'x' in col_name:
        col = col - landmarks_df['x1']
    elif 'y' in col_name:
        col = col - landmarks_df['y1']
    return col



def normalize_landmarks(col, landmarks_df):
    col_name = col.name
    if 'x' in col_name:
        col = col / landmarks_df['x13']
    elif 'y' in col_name:
        col = col / landmarks_df['y13']
    return col





@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)



@app.get("/")
def home():
    http_requests_total.labels(method="GET", endpoint="/").inc()
    return {"message": "Welcome to Home"}



@app.get("/health")
def check_health():
    http_requests_total.labels(method="GET", endpoint="/health").inc()
    return {"status": "healthy"}





@app.post("/predict")
async def predict(request: Request):

    http_requests_total.labels(method="POST", endpoint="/predict").inc()

    with inference_time_metric.time():
        try:
            start_time = time.time()

            data = await request.json()
            flat = data.get("landmarks")

            if not flat or len(flat) != 63:
                data_input_errors.inc()
                logging.warning("Data-related anomaly: Incorrect input size")
                return JSONResponse(status_code=400, content={"error": "Invalid number of landmarks"})

            # Preprocess data
            df = pd.DataFrame([flat], columns=feature_names)
            df_centered = df.apply(lambda col: center_landmarks(col, df), axis=0)
            df_normalized = df_centered.apply(lambda col: normalize_landmarks(col, df), axis=0)

            # Model prediction
            predictions = [
                svm_rbf_model.predict(df_normalized)[0],
                svm_poly_model.predict(df_normalized)[0],
                rf_model.predict(df_normalized)[0],
                xgb_label_encoder.inverse_transform([xgb_model.predict(df_normalized)[0]])[0]
            ]

            print("Models Predictions:", predictions)

            # Voting mechanism (choose the most common prediction)
            try:
                final_prediction = statistics.mode(predictions)
            except statistics.StatisticsError:
                final_prediction = predictions[-1]  # Choose XGB if there's a tie

            logging.info(f"Model-related: Inference latency = {round(time.time() - start_time, 4)} sec")

            return {"label": final_prediction}

        except Exception as e:
            model_failures.inc()
            logging.error(f"Prediction error: {str(e)}")
            return JSONResponse(status_code=500, content={"error": "Internal server error"})





# if __name__ == "__main__":
#     uvicorn.run("apis:app", host="0.0.0.0", port=8000, reload=False)

