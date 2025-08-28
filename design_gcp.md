# GCP Design

In order to save customer data, train the model, run inferences and monitor ML data distribution shift, I present the following architecture





1. **Data Storage** -> With **BQ**, we can continuously save and query customer data (not only for the ML Model training but also for independent data analyses).

2. **ML Experimentation and CI/CD** -> Once Customer data is saved in BQ, we could leverage **Vertex AI** for ML experimentation. **Cloud Build** (analogous to github actions) to automate the training and inference process, as well as save the output artifacts in artifact registry, in general to automate the CI/CD pipeline, saving.

3. **MLOps monitoring** --> Regarding the processing, we would run the images using **GCP Cloud Run** and finally track model performance using **Cloud Monitoring**.


