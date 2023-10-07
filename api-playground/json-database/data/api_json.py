from flask import Flask
from flask_restful import Resource, Api, reqparse, abort

import json

app = Flask("DatabaseAPI")
api = Api(app)


############################################
# Placeholder for a database
# Read the database from a json file
with open('data/database.json', 'r') as f:
  database = json.load(f)

# Save the database as a json file
def write_json_to_file(_database, filename='data/database.json'):
  # instead of passing database as an argument
  # global database 
  # sort by date first, dictionary comprehension
  database = {k: d for k, d in sorted(_database.items(), key=lambda data: data[1]['uploadDate'])}
  with open(filename, 'w') as f:
    json.dump(_database, f)
  return database



############################################
# Create a parser to validate the data
# (for PUT and POST request to add data to the database)
parser = reqparse.RequestParser()
parser.add_argument('title', type=str, help='Title of the data', required=True, 
  # per default is 'json'
  # but we want to parse the data 'form' style
  # {data_id: database[data_id]}
  location='form', 
)
parser.add_argument('uploadDate', type=int, help='Upload date of the data', required=False, location='form')



############################################
# Create resources
# multiple methods for one URL

# e.g. http://localhost:5000/database/data1
class Database(Resource):
  # GET data
  # e.g. http://localhost:5000/database/data1
  def get(self, data_id=None):
    # if data_id is None, return all data
    if data_id is None:
      return database
    return database[data_id]

  # add new data to database
  # but data_id is specified by url
  # e.g. curl http://localhost:5000/database/data100 -X PUT -d "title=Hello 100"
  def put(self, data_id):
    args = parser.parse_args()
    # database[data_id] =  {'title': args['title'], 'uploadDate': args['uploadDate']}
    database[data_id] = args
    write_json_to_file(database)
    # 201 is the status code for created
    return {data_id: database[data_id]}, 201

  # delete data from database
  # e.g. curl http://localhost:5000/database/data99 -X DELETE
  def delete(self, data_id):
    if data_id not in database:
      abort(404, message=f"Data {data_id} not found")
    del database[data_id]
    write_json_to_file(database)
    return '', 204


# e.g. http://localhost:5000/data
class DataSchedule(Resource):
  # GET data
  # e.g. http://localhost:5000/data
  def get(self):
    return database

  # add new data to database
  # without having to specify the data_id
  # e.g. curl http://localhost:5000/data -X POST -d "title=Hello last"
  # e.g. curl http://localhost:5000/data -X POST -d "title=Hello last" -d "uploadDate=20220101"
  def post(self):
    args = parser.parse_args()
    new_data = {'title': args['title'], 'uploadDate': args['uploadDate']}
    # new_data_id = max(int(v.lstrip('data')) for v in database.keys()) + 1
    new_data_id = f"data{len(database) + 1}"
    database[new_data_id] = new_data
    write_json_to_file(database)
    return database[new_data_id], 201



############################################
# Add the resources to the API with URLs
api.add_resource(Database, "/database/<string:data_id>")
api.add_resource(DataSchedule, "/data")


if __name__ == "__main__":
  # set debug=False for production
  app.run(debug=True)