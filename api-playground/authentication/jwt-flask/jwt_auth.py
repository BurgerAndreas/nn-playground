# to encode and decode JWT tokens
# conda install -c carta jwt
# conda install -c conda-forge pyjwt
import jwt
# for status codes, dependency injection
from fastapi import HTTPException, Security
# for
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
# for hashing passwords
# conda install -c conda-forge passlib
from passlib.context import CryptContext
# to set expiration time for jwt token
from datetime import datetime, timedelta


# 
class AuthHandler():
  # 
  security = HTTPBearer()
  # password hasher
  pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
  secret_key = "my_secret_key_to_encode_and_decode_my_jwt_token"

  # get plain password and return hashed password
  def get_password_hash(self, password):
    return self.pwd_context.hash(password)
  
  # get plain password and hashed password and return True if they match
  def verify_password(self, plain_password, hashed_password):
    return self.pwd_context.verify(plain_password, hashed_password)
  
  # get user id and return encoded jwt token
  def encode_token(self, user_id):
    # expiration time, current time, and user id
    payload = {
      "exp": datetime.utcnow() + timedelta(days=1, minutes=30),
      "iat": datetime.utcnow(),
      "sub": user_id
    }
    # return encoded jwt token
    return jwt.encode(payload, self.secret_key, algorithm="HS256")
  
  # get encoded jwt token and return decoded jwt token
  def decode_token(self, token):
    # try to decode jwt token
    try:
      payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
      return payload["sub"]
    # if jwt token is invalid
    except jwt.ExpiredSignatureError:
      raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
      raise HTTPException(status_code=401, detail="Invalid token")
  
  # Dependency injection wrapper
  def auth_wrapper(self, auth: HTTPAuthorizationCredentials = Security(security)):
    return self.decode_token(auth.credentials)
  
  