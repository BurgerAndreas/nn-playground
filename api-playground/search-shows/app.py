from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_session import Session
import sqlite3
import json
import sys

from sql_functions import dict_factory, sql_data_to_list_of_dicts

app = Flask(__name__)


# for API demonstration purposes
# search a database of shows
@app.route('/')
def index():
  return redirect("/search_database")

# HTML page
@app.route("/search_database")
def search_database():
  return render_template("search_database.html")

# API request
@app.route("/search")
def search():
  q = request.args.get("q")

  db_con = sqlite3.connect("data/shows.db", check_same_thread=False)
  db_con.row_factory = dict_factory # return a list of dictionaries
  db_cur = db_con.cursor()

  if q:
    q = "%" + q + "%"
    res = db_cur.execute("SELECT * FROM shows WHERE title LIKE ? LIMIT 50", (q,))
    shows = res.fetchall()
  else:
    shows = []

  db_con.close()

  # Alternative: 
  # use json instead of jsonify and dict_factory
  # return json.dumps(shows)

  return jsonify(shows)