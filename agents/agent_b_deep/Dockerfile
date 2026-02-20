FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir .
ENV PORT=8082
EXPOSE 8082
CMD ["python", "-m", "agents.agent_b_deep.service"]
