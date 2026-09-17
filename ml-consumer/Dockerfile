FROM python:3.11-slim

WORKDIR /app

COPY ml-consumer/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY kafka_ml_consumer.py .

COPY data ./data
COPY model ./model

CMD ["python", "kafka_ml_consumer.py"]
