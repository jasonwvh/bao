FROM python:3.11-slim
WORKDIR /app
COPY requirements.lock.txt /app/requirements.lock.txt
RUN pip install --no-cache-dir -r /app/requirements.lock.txt
COPY agents /app/agents
ENV PORT=8081
EXPOSE 8081
CMD ["python", "-m", "agents.ocsvm.service"]
