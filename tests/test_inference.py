# TODO: Boot API, call /predict using tests/sample.json

import pytest
from fastapi.testclient import TestClient
import json
from src.app import app  # Importamos la app de FastAPI

# 1. Creamos un cliente de prueba
# Este cliente simula una API en ejecución para que podamos hacerle peticiones.
client = TestClient(app)

# 2. Cargamos y preparamos los datos de muestra
# Leemos el archivo JSON que proporcionaste.
with open("tests/sample.json", "r") as f:
    sample_data = json.load(f)

# 3. Definimos la función de prueba
def test_predict_endpoint_returns_valid_prediction():
    """
    Prueba que el endpoint /predict procesa los datos y devuelve
    una predicción válida para cada muestra.
    """
    # Iteramos sobre cada uno de los dos clientes de ejemplo
    for customer_data in sample_data:
        # Hacemos una petición POST al endpoint /predict con los datos del cliente
        response = client.post("/predict/", json=customer_data)
        
        # 4. Verificamos la respuesta
        
        # El código de estado debe ser 200 (OK)
        assert response.status_code == 200, f"La API falló con los datos: {customer_data}"
        
        response_json = response.json()
        
        # La respuesta debe contener las claves 'churn_class' y 'churn_likelihood'
        assert "churn_class" in response_json
        assert "churn_likelihood" in response_json
        
        # La probabilidad ('churn_likelihood') debe ser un número entre 0 y 1
        assert 0.0 <= response_json["churn_likelihood"] <= 1.0
        
        # La clase ('churn_class') debe ser 0 o 1
        assert response_json["churn_class"] in [0, 1]

def test_health_check_returns_ok():
    """Prueba que el endpoint /health funciona correctamente."""
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
