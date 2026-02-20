FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir .
ENV PORT=8081
EXPOSE 8081
CMD ["python", "-m", "agents.agent_a_lightweight.service"]
