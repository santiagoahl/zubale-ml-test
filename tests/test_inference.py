# TODO: Boot API, call /predict using tests/sample.json

import pytest
from fastapi.testclient import TestClient
import json
from src.app import app  # Import the FastAPI app

INFERENCE_SAMPLE_FILE = "tests/sample.json"

# 1. Create testing client for inference
client = TestClient(app)

# 2. Load sample data
# Read JSON file
with open(INFERENCE_SAMPLE_FILE, "r") as f:
    sample_data = json.load(f)

# 3. Define the test function
def test_predict_endpoint_returns_valid_prediction():
    """
    Test whether /predict processes data and returns predictions
    It is expected a prediction for each sample
    """
    for customer_data in sample_data:
        
        response = client.post("/predict/", json=customer_data)
        
        # 4. Verify response
        assert response.status_code == 200, f"API failed processing customer data: {customer_data}"
        
        response_json = response.json()
        
        # The response must contain the keys 'churn_class' and 'churn_likelihood'
        assert "churn_class" in response_json
        assert "churn_likelihood" in response_json
        
        # The probability ('churn_likelihood') must be a number between 0 and 1
        assert 0.0 <= response_json["churn_likelihood"] <= 1.0
        
        # The class ('churn_class') must be 0 or 1
        assert response_json["churn_class"] in [0, 1]

def test_health_check_returns_ok():
    """Test /health endpoint"""
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}