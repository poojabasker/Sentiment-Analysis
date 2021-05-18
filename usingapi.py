import requests
import json

data={"Text":"bad"}

data=json.dumps(data)

#print(type(data))
#print(data)

API='http://127.0.0.1:5000/'

send_request = requests.post(API, data)
#print(send_request)

#print(send_request.json())