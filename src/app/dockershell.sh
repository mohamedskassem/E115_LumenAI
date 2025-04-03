#!/bin/sh

IMAGE_NAME="text-to-sql-app"

echo "Building Docker image '${IMAGE_NAME}'..."
docker build -t ${IMAGE_NAME} .

if [ $? -ne 0 ]; then
    echo "Docker build failed. Please check the Dockerfile and try again."
    exit 1
fi

echo "Stopping and removing existing container named 'lumenai' (if any)..."
docker stop lumenai > /dev/null 2>&1 || true
docker rm lumenai > /dev/null 2>&1 || true

echo "Running the Docker container '${IMAGE_NAME}' with name 'lumenai'..."
echo "Access the application at http://localhost:5001"
# Run the container in interactive mode, mapping port 5000, and mounting volumes
docker run -it --name lumenai \
    -p 5001:5000 \
    -v ./output:/app/output \
    -v ./secrets:/app/secrets \
    -e GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/google_api_key.json \
    ${IMAGE_NAME}

if [ $? -ne 0 ]; then
    echo "Docker run failed. Please check the container logs."
    exit 1
fi
