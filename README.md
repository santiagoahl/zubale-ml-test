<h1 align="center">

  🖥️ Mini-Prod ML Challenge
  <br>
</h1>

<h4 align="center">
  This repository is part of a technical test @ Zubale for a <b>Machine Learning Engineer</b> position.  
  The challenge is to code, build, test, deploy and monitor a <b>Customer Churn Predictive Model</b> using modern MLOps tools such as <b>Python, Pytest, FastAPI, Docker and Google Cloud Platform (GCP)</b>.
</h4>

<p align="center">
  <a href="https://www.python.org/" target="_blank"><img alt="python" src="https://img.shields.io/badge/Python-100000?style=for-the-badge&logo=python&logoColor=3776AB&labelColor=FFFFFF&color=3776AB"/></a>
  <a href="https://fastapi.tiangolo.com/" target="_blank"><img alt="fastapi" src="https://img.shields.io/badge/FastAPI-100000?style=for-the-badge&logo=fastapi&logoColor=009889&labelColor=FFFFFF&color=009889"/></a>
  <a href="https://cloud.google.com" target="_blank"><img alt="gcp" src="https://img.shields.io/badge/Google_Cloud-100000?style=for-the-badge&logo=google-cloud&logoColor=4285F4&labelColor=FFFFFF&color=4285F4"/></a>
  <a href="https://www.docker.com/" target="_blank"><img alt="docker" src="https://img.shields.io/badge/Docker-100000?style=for-the-badge&logo=docker&logoColor=218bea&labelColor=FFFFFF&color=FFFFFF"/></a>
  <a href="https://python-poetry.org/" target="_blank"><img alt="poetry" src="https://img.shields.io/badge/Poetry-100000?style=for-the-badge&logo=poetry&logoColor=60A5FA&labelColor=FFFFFF&color=2563EB"/></a>
  <a href="https://scikit-learn.org/" target="_blank"><img alt="scikit-learn" src="https://img.shields.io/badge/Scikit_Learn-100000?style=for-the-badge&logo=scikit-learn&logoColor=FFFFFF&labelColor=FF4400&color=0563FF"/></a>
  <a href="https://pytest.org/" target="_blank"><img alt="pytest" src="https://img.shields.io/badge/Pytest-100000?style=for-the-badge&logo=pytest&logoColor=0A9EDC&labelColor=FFFFFF&color=0A9EDC"/></a>
</p>

<p align="center">
  <a href="#gcp-architecture">GCP Architecture</a> •
  <a href="#🛠️-setup">Setup</a> •
  <a href="#🔁-run-the-mlops-flow">Run Flow</a> •
  <a href="#📄-license">License</a> 
</p>

---



--- ## GCP Architecture The deployment is thought to be built using GCP, with services such as VertexAI or BigQuery. Details are saved in the gcp_design.md ![GCP Architecture](https://raw.githubusercontent.com/santiagoahl/zubale-ml-test/main/GCP%20Architecture%20for%20Zubale%20Product.drawio.png) ## **🛠️ Setup** This project uses [__Poetry__](__https://python-poetry.org/__) for dependency management. ### **1. Clone the repository**
bash
git clone git@github.com:santiagoahl/zubale-ml-test.git
cd zubale-ml-test
### 2. Install dependencies Make sure you have Poetry installed. If not, simply run
bash
python -m pip install --user pipx
python -m pipx ensurepath
pipx install poetry
poetry install
That will install all necessary dependencies ### **3. Activate the environment**
bash
poetry shell
That will create a virtual environment for the project --- ## **🔁 Run the MLOps flow**
bash
# Train model
python -m src.train --data data/customer_churn_synth.csv --outdir
artifacts/  # Check `artifacts/` for the training output artifacts (feature pipeline, model as well as SHAP values)

# Run API
uvicorn src.app:app  # See `docs/` endpoint for trying out predictions

# Test Training and inference pipelines
python -m pytest
# Run AI Monitor In order to run gpt-4o inside the AI monitor, you need to save the credential keys inside the .env file
bash
OPENAI_API_KEY=<your_key>
OPENAI_API_KEY=...
LANGCHAIN_API_KEY=...
LANGSMITH_API_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_HOST=...
--- ## **📄 License** MIT License. ---
