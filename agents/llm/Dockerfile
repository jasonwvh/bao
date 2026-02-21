FROM python:3.11-slim
WORKDIR /app
COPY agents/llm /app
ENV PORT=8084
EXPOSE 8084
CMD ["python", "-m", "service"]
