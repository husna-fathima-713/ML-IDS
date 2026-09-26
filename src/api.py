import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


MODEL_DIR = "results/model"

THRESHOLD = 0.5

MODEL_PATH = f"{MODEL_DIR}/random_forest.joblib"
PREPROCESSOR_PATH = f"{MODEL_DIR}/preprocessor.joblib"


# Load trained model and preprocessing pipeline
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


app = FastAPI(
    title="ML Intrusion Detection System",
    description="Real-time ML-based network intrusion detection prototype",
    version="1.0"
)


# Allow the dashboard running on port 5500
# to communicate with the API running on port 8000.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrafficData(BaseModel):

    duration: int

    protocol_type: str

    service: str

    flag: str

    src_bytes: int

    dst_bytes: int

    land: int

    wrong_fragment: int

    urgent: int

    hot: int

    num_failed_logins: int

    logged_in: int

    num_compromised: int

    root_shell: int

    su_attempted: int

    num_root: int

    num_file_creations: int

    num_shells: int

    num_access_files: int

    num_outbound_cmds: int

    is_host_login: int

    is_guest_login: int

    count: int

    srv_count: int

    serror_rate: float

    srv_serror_rate: float

    rerror_rate: float

    srv_rerror_rate: float

    same_srv_rate: float

    diff_srv_rate: float

    srv_diff_host_rate: float

    dst_host_count: int

    dst_host_srv_count: int

    dst_host_same_srv_rate: float

    dst_host_diff_srv_rate: float

    dst_host_same_src_port_rate: float

    dst_host_srv_diff_host_rate: float

    dst_host_serror_rate: float

    dst_host_srv_serror_rate: float

    dst_host_rerror_rate: float

    dst_host_srv_rerror_rate: float


@app.get("/")
def root():

    return {
        "service": "ML Intrusion Detection System",
        "status": "running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "Random Forest",
        "threshold": THRESHOLD
    }


@app.post("/predict")
def predict(data: TrafficData):

    input_data = pd.DataFrame([
        data.model_dump()
    ])

    processed_data = preprocessor.transform(
        input_data
    )

    probability = float(
        model.predict_proba(
            processed_data
        )[0][1]
    )

    prediction = int(
        probability >= THRESHOLD
    )

    decision = (
        "BLOCK"
        if prediction == 1
        else "ALLOW"
    )

    return {
        "prediction": prediction,
        "classification": (
            "Attack"
            if prediction == 1
            else "Normal"
        ),
        "risk_score": round(
            probability,
            4
        ),
        "threshold": THRESHOLD,
        "decision": decision
    }