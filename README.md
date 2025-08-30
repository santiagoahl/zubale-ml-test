# **🖥️ Mini-Prod ML Challenge**

This Mini-Prod Challenges is part of a tech test @ Zubale for a ML Engineer postition. It is made to code, build, test, deploy and test a **Customer Churn Predictive Model** using Python, Pytest, FastAPI, Docker and GCP.

---
## GCP Architecture

![GCP Architecture](https://raw.githubusercontent.com/santiagoahl/zubale-ml-test/main/GCP%20Architecture%20for%20Zubale%20Product.drawio.png)

## **🛠️ Setup**

This project uses [__Poetry__](__https://python-poetry.org/__) for dependency management.

### **1. Clone the repository**

```bash
git clone git@github.com:santiagoahl/zubale-ml-test.git
cd zubale-ml-test
````

### 2. Install dependencies

Make sure you have Poetry installed. If not, simply run

```bash
python -m pip install --user pipx
python -m pipx ensurepath
pipx install poetry
poetry install
```
That will install all necessary dependencies


### **3. Activate the environment**


```bash
poetry shell
```
That will create a virtual environment for the project

---

## **🔁 Run the MLOps flow**


```bash
# Train model
python -m src.train --data data/customer_churn_synth.csv --outdir
artifacts/  # Check `artifacts/` for the training output artifacts (feature pipeline, model as well as SHAP values)

# Run API
uvicorn src.app:app  # See `docs/` endpoint for trying out predictions

# Test Training and inference pipelines
python -m pytest
```

# Run AI Monitor 
In order to run gpt-4o inside the AI monitor, you need to save the credential keys inside the .env file

```bash
OPENAI_API_KEY=<your_key>
OPENAI_API_KEY=...
LANGCHAIN_API_KEY=...
LANGSMITH_API_KEY=...
HF_TOKEN_CHAPPIE=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_HOST=...
```
---

## **📄 License**

MIT License.


--- 
