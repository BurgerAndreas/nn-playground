from fastapi import FastApi
from fastapi import HTTPException, Depends

from .jwt_auth import AuthHandler
from .schemas import AuthDetails

app = FastApi()

users = []

# create user
# curl --header "Content-Type: application/json" --request POST --data '{"username":"andreas","password":"secretpassword"}' localhost:8000/register
@app.post("/register", status_code=201)
def register(auth_details: AuthDetails):
  # check if username already exists
  if any(user["username"] == auth_details.username for user in users):
    return {"message": "Username already exists"}
  # hash password
  hashed_password = AuthHandler.get_password_hash(auth_details.password)
  # add user to users list
  users.append({
    "username": auth_details.username,
    "password": hashed_password
  })
  return 


# login user
# curl --header "Content-Type: application/json" --request POST --data '{"username":"andreas","password":"secretpassword"}' localhost:8000/login
@app.post("/login")
def login(auth_details: AuthDetails):
  # check if user exists
  if not any(user["username"] == auth_details.username for user in users):
    return {"message": "Username does not exist"}
  
  # check if password is correct
  user = next(user for user in users if user["username"] == auth_details.username)
  if not AuthHandler.verify_password(auth_details.password, user["password"]):
    raise HTTPException(status_code=401, detail="Incorrect password")
  else:
    # generate jwt token
    token = AuthHandler.encode_token(user["username"])
    return {"token": token}



@app.get("/unprotected")
def unprotected():
  return {"message": "Anyone can view this"}


# 
# curl --header 'Authorization: Bearer <token>' localhost:8000/protected
@app.get("/protected")
def protected(username=Depends(AuthHandler.auth_wrapper)):
  return {'name': 'username'}