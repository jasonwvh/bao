FROM python:3.11-slim
WORKDIR /app
COPY requirements.lock.txt /app/requirements.lock.txt
RUN pip install --no-cache-dir -r /app/requirements.lock.txt
COPY agents /app/agents
ENV PORT=8084
EXPOSE 8084
CMD ["python", "-m", "agents.wgan_gp.service"]
