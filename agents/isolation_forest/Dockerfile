FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy pandas scikit-learn
COPY agents/isolation_forest /app
ENV PORT=8081
EXPOSE 8081
CMD ["python", "-m", "service"]
