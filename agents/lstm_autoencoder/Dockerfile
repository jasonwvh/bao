FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy pandas torch fastapi uvicorn pydantic
COPY agents /app/agents
ENV PORT=8082
EXPOSE 8082
CMD ["python", "-m", "agents.lstm_autoencoder.service"]
