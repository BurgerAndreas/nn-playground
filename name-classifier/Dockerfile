# Build docker image
# docker build -f Dockerfile -t docker-clf-model . 
# Run docker image
# docker run docker-clf-model

FROM python:3.8-buster

#WORKDIR ../name-classifier
#VOLUME ../name-classifier

#RUN pip install --upgrade pip
RUN pip install numpy==1.23.1 pandas==1.4.3 flask==2.1.3 scikit-learn==1.1.1 matplotlib==3.5.1
RUN pip install tensorflow=2.8.2

COPY name_gender.csv ./name_gender.csv
COPY saved_models/ ./saved_models/
COPY templates/ ./templates/

COPY main.py ./main.py
COPY web_app.py ./web_app.py
COPY lstm.py ./lstm.py
COPY naive_bayes.py ./naive_bayes.py

RUN python3 main.py
RUN python3 web_app.py
