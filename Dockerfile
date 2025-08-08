# Base image
FROM registry.hub.docker.com/library/python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run Flask app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
