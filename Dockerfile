FROM python:3.7-slim

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

# Copy app
COPY . .

# Environment variables
ENV FLASK_APP=WebbApp/app.py
ENV PORT=5000

# Expose port
EXPOSE $PORT

# Run Flask with dynamic port from Render
CMD cd /app/WebbApp && python app.py