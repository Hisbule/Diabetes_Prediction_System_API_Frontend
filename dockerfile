FROM python:3.9-slim

WORKDIR /app


COPY best_diabetes_model.joblib /app/best_diabetes_model.joblib
COPY scaler.joblib /app/scaler.joblib   

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./Fast_api ./Fast_api
COPY ./model ./model

CMD ["uvicorn", "Fast_api.main:app", "--host","0.0.0.0" ,"--port", "8080", "--reload"]