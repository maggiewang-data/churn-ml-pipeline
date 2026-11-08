FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY data/ data/
# Train inside the image so the artifact always matches the library versions
RUN cd src && python -m churn.train
EXPOSE 8000
CMD ["uvicorn", "churn.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
