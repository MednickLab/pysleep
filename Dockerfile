FROM python:3.6.8-slim
WORKDIR /pysleep
copy . .
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get -qq update && apt-get -qq install -y \
    build-essential \
    curl \
    unzip \
    wget \
    xorg \
    git \
    tzdata
ENV TZ America/Los_Angeles
RUN pip install -r requirements.txt && pip install pyedflib
CMD ["pytest","-v","tests"]

