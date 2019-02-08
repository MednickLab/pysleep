FROM python:3.6.8-slim
WORKDIR /pysleep
COPY . .
RUN apt-get -y update && apt-get -y upgrade
RUN pip install -r requirements.txt


