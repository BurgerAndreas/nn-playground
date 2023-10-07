from pydantic import BaseModel


# Schema to serialize and deserialize login data
# What is to be included in a login request
class AuthDetails(BaseModel):
  username: str
  password: str

