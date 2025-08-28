# TODO: .
# Endpoints: GET /health, POST /predict

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from .io_schemas import PredictModel
import joblib
import pandas as pd

# Get churn model

churn_model = joblib.load("artifacts/model.pkl")
feature_pipeline = joblib.load("artifacts/feature_pipeline.pkl")

app = FastAPI()


@app.get("/health/") 
def get_health() -> dict[str, str]:
    """
    Path Operation to monitor churn model performance.

    Parameters
    ----------
    None
    
    Returns:
        dict: Status Code
    """
    return {"status":"ok"}


@app.post("/predict/")
def post_predict(customer_data: PredictModel):
    """
    Path Operation to consume Churn model and predict.

    Parameters
    ----------
    customer_data : PredictModel
        Customer info used by the model to predict churn
    
    Returns:
        dict: Churn category (0 = Not likely to churn, 1 = Likely to churn) and churn estimated probability
    """
    try:
        customer_data = pd.DataFrame([jsonable_encoder(customer_data)])
        customer_data = feature_pipeline.transform(customer_data)
        churn_class = int(churn_model.predict(customer_data)[0])
        churn_likelihood = float(churn_model.predict_proba(customer_data)[0][1]  )
        return {
            "churn_class": churn_class,
            "churn_likelihood": churn_likelihood,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))