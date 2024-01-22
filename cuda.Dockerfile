FROM nvidia/cuda:12.3.1-base-ubuntu20.04

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils && \
    python3.10 -m ensurepip && \
    python3.10 -m pip install --upgrade pip

RUN ln -s /usr/bin/python3.10 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/local/bin/python

WORKDIR /app

COPY ./requirements.txt ./
COPY ./src ./

RUN pip install --no-cache-dir -r requirements.txt

CMD ["sh"]