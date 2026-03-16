FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app
COPY . .

# Environment variables
ENV FLASK_APP=WebbApp/app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Expose ports for Flask (5000) and Jupyter (8888)
EXPOSE 5000 8888

# The default command runs Flask; Jupyter can be run via docker-compose command override
CMD ["flask", "run"]
