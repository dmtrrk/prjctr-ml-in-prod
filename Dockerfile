FROM python:3.10.11-slim-bullseye

WORKDIR /app

COPY ./requirements.txt ./
COPY ./src ./

RUN pip install --no-cache-dir -r requirements.txt

CMD ["sh"]