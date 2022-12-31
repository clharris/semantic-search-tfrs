FROM nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu18.04

RUN apt-get update && \
    apt-get install -y vim \
    curl \
    zip \
    && apt-get install -y python3.8 python3.8-distutils

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

WORKDIR /home/root
RUN mkdir data/
COPY data/training_and_product_data.zip data/training_and_product_data.zip
RUN cd data/ && unzip training_and_product_data.zip && rm training_and_product_data.zip

WORKDIR /home/root
COPY src/*.py src/
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade -r requirements.txt
