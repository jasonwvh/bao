FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir .
ENV PORT=8083
EXPOSE 8083
CMD ["python", "-m", "agents.agent_c_contextual.service"]
