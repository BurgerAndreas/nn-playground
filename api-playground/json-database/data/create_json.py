from api_json import write_json_to_file

# Placeholder for the database in api.py
# Create once and then read from file
def create_database_json():
  database_fake = {"data1": {'title': 'Hello 1', 'uploadDate': 20230131}}
  for _ in range(2, 100):
    database_fake[f"data{_}"] = {'title': f"Hello {_}", 'uploadDate': 20230131}
  # print(database_fake)
  return database_fake
# database = create_database_json()
# write_json_to_file(database)