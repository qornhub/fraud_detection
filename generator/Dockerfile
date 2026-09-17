FROM python:3.11-slim

WORKDIR /app

COPY generator/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY transaction_generator.py .

CMD ["python", "transaction_generator.py"]
