FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .


COPY static ./static

COPY random_forest_model.joblib .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
