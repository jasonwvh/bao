FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy pandas torch
COPY agents/autoencoder /app
ENV PORT=8082
EXPOSE 8082
CMD ["python", "-m", "service"]
