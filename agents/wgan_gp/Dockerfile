FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy torch scikit-learn fastapi uvicorn pydantic
COPY agents /app/agents
ENV PORT=8084
EXPOSE 8084
CMD ["python", "-m", "agents.wgan_gp.service"]
