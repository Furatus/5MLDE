FROM ghcr.io/mlflow/mlflow:v3.5.1

WORKDIR /mlflow/
EXPOSE 5000

ENV BACKEND_URI postgresql://winedefaultuser:Gmkb1zRq1JNIa8nb@db:5432/mlflowdb
ENV ARTIFACT_ROOT http://localhost:5000
ENV ARTIFACT_DESTINATION http://localhost:5000
ENV MLFLOW_SERVER_ALLOWED_HOSTS *

RUN pip install psycopg2-binary 

CMD mlflow server --backend-store-uri ${BACKEND_URI} --host 0.0.0.0 --port 5000