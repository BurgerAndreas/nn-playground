# Name Classifier
Classify names by gender. With a web-app and docker. \
Using a simple Naive Bayes, or a LSTM model.

## Quick Start
### Using Docker
$ docker build -f Dockerfile -t docker-clf-model . \
$ docker run docker-clf-model 

### Using Conda
$ conda env create -f conda-clf-model.yml \
$ conda activate conda-clf-model \
$ python3 main.py 

$ python3 web_app.py \
returns: Running on http://127.0.0.1:5000 \
open http://127.0.0.1:5000 in your browser 


# Explanation
As often in real life, I only had a few hours for the project. \
So I looked into how others solved similar problems. 

My first approach was simple and fast. \
It's a Naive Bayes model. \
It's not very accurate (75% test accuracy). \
(Try: main.py, Implementation: naive_bayes.py) 

My second approach was a bit bit more sophisticated. \
It's a bidirectional LSTM. \
It takes a lot longer to train but is pretty accurate (> 90% test accuracy). \
(Try: main.py or web_app.py, Implementation: lstm.py) 


### Requirements.txt and Conda
requirements.txt was generated using \
$ conda list -e > requirements.txt 

conda-clf-model.yml was generated using \
$ conda env export > conda-clf-model.yml

