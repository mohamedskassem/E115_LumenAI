FROM python:3.9-slim-buster

WORKDIR /app

COPY Pipfile Pipfile.lock ./

RUN pip install pipenv && pipenv install --system

COPY . .

RUN python download_model.py

CMD ["python", "app.py"]
