FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy pandas scikit-learn fastapi uvicorn pydantic
COPY agents /app/agents
ENV PORT=8081
EXPOSE 8081
CMD ["python", "-m", "agents.ocsvm.service"]
