# Use a slim and official Python image
FROM python:3.10-slim

# Install git and git-lfs for handling model files
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy and install dependencies first, to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the models directory first and ensure LFS files are pulled
COPY models/ models/

# Alternative: If the above doesn't work, copy the model file explicitly
# Make sure the model file is actually committed to your repo
# You can also download it from a URL if needed:
# RUN wget -O models/model.tflite "YOUR_MODEL_DOWNLOAD_URL"

# Verify the model file exists and has the correct size
RUN ls -la models/ && \
    if [ -f "models/model.tflite" ]; then \
        echo "Model file found, size: $(stat -c%s models/model.tflite) bytes"; \
        if [ $(stat -c%s models/model.tflite) -lt 1000000 ]; then \
            echo "WARNING: Model file is smaller than 1MB, might be a Git LFS pointer"; \
            cat models/model.tflite; \
        fi; \
    else \
        echo "ERROR: Model file not found!"; \
        exit 1; \
    fi

# Copy the rest of your application files into the container
COPY . .

# Create necessary directories
RUN mkdir -p static/uploads logs

# Tell Docker that the container listens on port 5001
EXPOSE 5001

# The command to run your application using Gunicorn
CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:5001", "wsgi:app"]