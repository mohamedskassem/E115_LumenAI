FROM python:3.9-slim

WORKDIR /app

# Copy Pipfile and Pipfile.lock first to leverage Docker cache.
COPY Pipfile Pipfile.lock ./

# Install pipenv and then install dependencies into the system environment.
RUN pip install pipenv && pipenv install --system

# Copy the rest of your application code.
COPY . .

# Expose any ports if needed (optional)
# EXPOSE 8080

# Run the application.
CMD ["python", "app.py"]
