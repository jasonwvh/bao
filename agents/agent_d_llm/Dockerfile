FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir .
ENV PORT=8084
EXPOSE 8084
CMD ["python", "-m", "agents.agent_d_llm.service"]
