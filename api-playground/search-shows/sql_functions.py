import sqlite3
import json

# Usage:
# con.row_factory = dict_factory
# https://docs.python.org/3/library/sqlite3.html#sqlite3-howto-row-factory
def dict_factory(cursor, row):
  """Modifier for row_factory property to return a list of dictionaries instead of tuples."""
  d = {}
  for idx, col in enumerate(cursor.description):
    d[col[0]] = row[idx]
  return d


# Usage:
# QUERY = "SELECT * FROM data"
# returned_data = sql_data_to_list_of_dicts(SAMPLE_DB, QUERY)
# print(returned_data)
def sql_data_to_list_of_dicts(path_to_db, select_query):
  """Returns data from an SQL query as a list of dicts."""
  try:
    con = sqlite3.connect(path_to_db)
    con.row_factory = sqlite3.Row
    things = con.execute(select_query).fetchall()
    unpacked = [{k: item[k] for k in item.keys()} for item in things]
    return unpacked
  except Exception as e:
    print(f"Failed to execute. Query: {select_query}\n with error:\n{e}")
    return []
  finally:
    con.close()


# Usage:

def get_my_jsonified_data(path_to_db, select_query):
  """Returns data from an SQL query as a JSON string."""
  with sqlite3.connect(path_to_db) as con:
    cursor = con.cursor()
    cursor.execute(select_query)
    data = cursor.fetchall()
    return json.dumps(data)
