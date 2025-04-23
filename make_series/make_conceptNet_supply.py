import requests
import pprint
obj = requests.get('http://api.conceptnet.io/c/en/next').json()
print(obj.keys())

print(len(obj['edges']))

print(obj['@id'])
pprint.pprint(obj['edges'][0:3])