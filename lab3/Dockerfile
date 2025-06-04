FROM python:3.10-slim

ARG GIT_SHA
LABEL git_sha=$GIT_SHA

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py .

COPY ../lab1/model.pkl .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]