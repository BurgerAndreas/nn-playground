from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_session import Session
import sqlite3
import json
import sys

################################################
# To start server
# flask run

# To print debug info into console server side
# print('Hello Console', file=sys.stdout, flush=True)

# Check database using sqlite3
# sqlite data/sport.db
# .tables
# .schema registered_user
# SELECT * FROM registered_user;
################################################


app = Flask(__name__)

# session (=cookies) for login
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = 'filesystem'
Session(app)


# Capitalized variables are global variables
SPORTS = ['Basketball', 'Volleyball', 'Ultimate Frisbee', 'Football', 'Swimming', 'Karate']


@app.route('/')
def index():
  # login first
  if not session.get("name"):
    return redirect("/login")
  return render_template('index.html', sports=SPORTS)


# login
@app.route('/login', methods=['GET', 'POST'])
def login():
  # set name cookie
  if request.method == "POST":
    session["name"] = request.form.get("loginname")
    # proceed to index (sport registration page)
    return redirect("/")
  # if get request, render login page
  return render_template('login.html')


# logout
@app.route('/logout')
def logout():
  # remove name cookie
  session["name"] = None
  return redirect("/")


@app.route('/registersport', methods=['POST'])
def registersport():
  # username = request.form.get("username")
  username = session.get("name")
  sport = request.form.get("sport", SPORTS[0])

  # database
  db_con = sqlite3.connect("data/sport.db", check_same_thread=False)
  db_cur = db_con.cursor()

  # Server side validation
  if sport not in SPORTS:
    return render_template("error_sport.html", errormsg='invalid sport')
  if username:
    res = db_cur.execute("SELECT user_name, user_sport FROM registered_user WHERE user_name = ?", (username,))
    found_user_sport = res.fetchone()
    if found_user_sport:
      return render_template("error_sport.html", errormsg='User "{}" is already registered for "{}"'.format(str(found_user_sport[0]), str(found_user_sport[1])))
  else:
    return render_template("error_sport.html", errormsg='username is required')

  # Save data
  # Question marks against SQL injection
  db_cur.execute("""
    INSERT INTO registered_user (user_name, user_sport) VALUES(?, ?)
  """, (username, sport))
  db_con.commit()
  db_con.close()

  # return redirect(url_for('index'))
  # return redirect('/showregistrants')
  return render_template("success_sport.html", username=username, sport=sport)



@app.route('/showregistrants')
def showregistrants():

  # get registrants from database
  db_con = sqlite3.connect("data/sport.db", check_same_thread=False)

  # to always return  a kind of list of dictionaries, 
  # set row_factory property of connection 
  # default is a list of tuples
  db_con.row_factory = sqlite3.Row

  db_cur = db_con.cursor()

  res = db_cur.execute("SELECT user_name, user_sport FROM registered_user")
  registrants = res.fetchall()
  db_con.close()
  
  # Not working: make into list of dictionaries
  # registrants_dict = [{k: _row[k] for k in _row.keys()} for _row in registrants]
  # Not working: make into list of dictionaries
  # registrants_dict = [dict(_row) for _row in registrants]

  return render_template('showregistrants.html', registrants=registrants)



@app.route('/deregister', methods=['POST'])
def deregister():
  deregister_user_name = request.form.get("deregister_user_name")
  deregister_user_sport = request.form.get("deregister_user_sport")

  if deregister_user_name:
    db_con = sqlite3.connect("data/sport.db", check_same_thread=False)
    db_cur = db_con.cursor()

    db_cur.execute("""
      DELETE FROM registered_user WHERE user_name = ?
    """, (deregister_user_name,))
    # Alternative: delete sport but remember user
    # have to change Server side validation & save data in registersport()
    # UPDATE registered_user SET user_sport = NULL WHERE user_name = ?

    db_con.commit()
    db_con.close()
  
  return redirect('/showregistrants')


if __name__ == '__main__':
  app.run(debug=True)