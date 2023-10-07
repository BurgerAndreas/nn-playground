# conda install -c conda-forge flask-login flask -y
from flask import Flask, redirect, url_for, session, request
from flask_login import LoginManager, login_required, login_user, logout_user, current_user
# conda install -c conda-forge flask-oauthlib
# conda install -c conda-forge Authlib Flask
from authlib.integrations.flask_client import OAuth
from datetime import timedelta
# load environment variables
from dotenv import load_dotenv
import os


app = Flask(__name__)

# Session config
load_dotenv()
app.secret_key = os.getenv("APP_SECRET_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
app.config['SESSION_COOKIE_NAME'] = 'google-login-session'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=5)


# oauth config
oauth = OAuth(app)
oauth.register(
  name='google',
  client_id=GOOGLE_CLIENT_ID,
  client_secret=GOOGLE_CLIENT_SECRET,
  access_token_url='https://accounts.google.com/o/oauth2/token',
  access_token_params=None,
  authorize_url='https://accounts.google.com/o/oauth2/auth',
  authorize_params=None,
  api_base_url='https://www.googleapis.com/oauth2/v1/',
  # scope is what user_info will be returned
  client_kwargs={'scope': 'openid email profile'},
)


@app.route('/')
def hello_user():
  return 'Hello World!'


@app.route('/user_area')
@login_required
def for_logged_in_users_only():
  email = dict(session).get('email', default=None)
  # email = dict(session)['profile']['email']
  return f'Hello {email}!'


@app.route('/login')
def login():
  google = oauth.create_client('google')
  redirect_uri = url_for('authorize', _external=True)
  return google.authorize_redirect(redirect_uri)


@app.route('/authorize')
def authorize():
  google = oauth.create_client('google')  # create the google oauth client
  token = google.authorize_access_token()  # Access token from google (needed to get user info)
  resp = google.get('userinfo', token=token)  # userinfo contains stuff u specificed in the scrope
  user_info = resp.json()
  user = oauth.google.userinfo()  # uses openid endpoint to fetch user info
  
  # Here you use the profile/user data 
  # maybe query your database to find/register the user
  # and set your own data in the session
  session['profile'] = user_info
  session.permanent = True  # make the session permanant so it keeps existing after browser gets closed
  
  return redirect('/user_area')


@app.route('/logout')
@login_required
def logout():
  for key in list(session.keys()):
    session.pop(key)
  return redirect('/')


if __name__ == '__main__':
  app.run(debug=True)