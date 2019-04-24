FROM python:3.6.8-slim
WORKDIR /pysleep
RUN apt-get -qq update && apt-get -qq install -y 
copy . .
RUN pip install -r requirements.txt
CMD ["pytest","-v","tests"]

