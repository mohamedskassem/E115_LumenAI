FROM python:3.9-slim

WORKDIR /app

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt ./

# Install dependencies from requirements.txt
# --no-cache-dir reduces layer size, --upgrade pip ensures latest pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code (respecting .dockerignore)
COPY . .

# Expose port 5000 for the Flask app
EXPOSE 5000

# Run the application.
CMD ["python", "server.py"]
