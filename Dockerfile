FROM python:3.6.8-slim
WORKDIR /pysleep
copy . .
RUN pip install -r requirements.txt && pip install pyedflib
CMD ["pytest","-v","tests"]

