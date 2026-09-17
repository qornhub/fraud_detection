FROM python:3.11-slim

WORKDIR /app

COPY dashboard_requirements.txt .

RUN pip install --no-cache-dir -r dashboard_requirements.txt

COPY dashboard.py .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
