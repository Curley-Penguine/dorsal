FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV DORSAL_LLM=offline DORSAL_EVIDENCE_DIR=/app/runs/live
EXPOSE 8000
CMD ["python", "-m", "dorsal", "serve", "--host", "0.0.0.0", "--port", "8000"]
