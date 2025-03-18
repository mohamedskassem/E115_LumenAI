# Use the official Python 3.9 slim image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy Pipfile and Pipfile.lock (if available) to install dependencies
COPY Pipfile Pipfile.lock* ./

# Install pipenv and project dependencies using the lock file
RUN pip install pipenv && pipenv sync --system

# Copy the data loader code into the container
COPY . .

# Define a volume to mount the local 'data' folder into the container at /data
VOLUME ["/data"]

# Run the data loader script
CMD ["python", "load_data.py"]
