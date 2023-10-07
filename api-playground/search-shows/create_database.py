import sqlite3

# only run once to setup database
# it's possible to connect to database globally by using check_same_thread=False
#  db_con = sqlite3.connect("data/sport.db", check_same_thread=False
# but it's probably better to connect to the database fresh in each method
def create_database():
  # Connects to or creates database
  db_con = sqlite3.connect("data/sport.db", check_same_thread=False)
  db_cur = db_con.cursor()
  # Create table if it doesn't exist # <column> <datatype> <constraint>
  db_cur.execute("""
    CREATE TABLE IF NOT EXISTS registered_user(
      user_id INTEGER PRIMARY KEY, 
      user_name TEXT UNIQUE, 
      user_sport TEXT
    )
  """)
  db_con.commit()
  db_con.close()
  return