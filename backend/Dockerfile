# Use an official Python base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean

# Set work directory
WORKDIR /app

# Copy backend code and requirements
COPY . /app
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install yt-dlp
RUN pip install yt-dlp

# Expose the backend port
EXPOSE 5000

# Set environment variable for Flask
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the backend
CMD ["python", "main.py"] 