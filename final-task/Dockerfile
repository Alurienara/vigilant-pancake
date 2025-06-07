FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "evaluate.py", "models/model.pkl", "data/prepared.csv"]