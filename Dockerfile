# Use a slim and official Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy and install dependencies first, to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application files into the container
COPY . .

# Tell Docker that the container listens on port 5001
EXPOSE 5001

# The command to run your application using Gunicorn
# We use wsgi:app because you have a wsgi.py file
CMD ["gunicorn", "--workers", "3", "--bind", "0.0.0.0:5001", "wsgi:app"]