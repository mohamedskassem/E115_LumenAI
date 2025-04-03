FROM python:3.9-slim

WORKDIR /app

# Copy Pipfile and Pipfile.lock first to leverage Docker cache.
COPY Pipfile Pipfile.lock ./

# Install pipenv and then install dependencies into the system environment.
RUN pip install pipenv && pipenv install --system

# Copy the rest of your application code.
COPY . .

# Expose port 5000 for the Flask app
EXPOSE 5000

# Run the application.
CMD ["python", "server.py"]
