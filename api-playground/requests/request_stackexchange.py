import requests
import json

response = requests.get('https://api.stackexchange.com/2.2/questions?order=desc&sort=activity&site=stackoverflow')

print(response.status_code)

# print(response.json())

for data in response.json()['items']:
  print(data['title'])
  print(data['link'])
  print(data['is_answered'])
  print(data['view_count'])
  print(data['answer_count'])
  print(data['score'])
