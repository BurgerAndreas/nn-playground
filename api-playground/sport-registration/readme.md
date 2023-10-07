# Features

- Flask
- Docker
- Forms (POST/GET)
- SQLite database
- Cookies (Session)

# Future

- CSS / React styling
- API that can access proper database
- Authentication (Login with password)


# Installation

## Conda

```
conda install -c conda-forge flask flask-restful flask-session flask-sqlalchemy -y
```

## Docker Container 

### Usage

```
docker build -t sportregistration .

docker run -it -p 5000:5000 -d sportregistration
``` 

Open browser to
http://localhost:5000

```
docker ps

docker stop <container id>

# Stop all containers
docker rm -f $(docker ps -aq)

# find what's using a port (like the new MacOS AirPlay)
sudo lsof -i -P -n | grep <port number>  
```

### Container was made using

```
conda install pip

pip freeze > requirements.txt
# or
pip list --format=freeze > requirements.txt
```