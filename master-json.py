'''
MASTERING JSON FILES WITH PYTHON
JSON (JavaScript Object Notation) has become the de facto standard for data interchange between systems due to its simplicity,
human-readability, and widespread support.
'''

import json
'''
JSON structure:
{
  "name": "Oscar",
  "surname": "Arauz",
  "age": 22,
  "city": "Seville",
  "profession": "Just a noob programmer",
  "proyects": [
    {
      "id": 1,
      "name": "ChatPDF",
      "lang": "Python"
    },
    {
      "id": 2,
      "name": "IRIS",
      "lang": "Python"
    }
  ]
}
'''

# Convert JSON string into Python data structure.
json_data = '{"name": "Óscar", "surname": "Arauz", "age": 22, "city": "Seville", "profession": "Dumb AI programmer"}'
data = json.loads(json_data)
print(data['name'])
print(data['surname'])
print(data['age'])
print(data['city'])
print(data['profession'])
print(data)

# Serializing Python Objects to JSON.
json_data = {"name": "Oscar",
             "surname": "Arauz",
             "age": 22,
             "city": "Seville",
             "profession": "Dumb AI programmer"}

#data = json.dumps(json_data)
#print(data)

# Reading and Writing Data from JSON files.
# Writing
with open('files/data.json', 'w') as file:
    json.dump(json_data, file)
# Reading
with open('files/data.json', 'r') as file:
    data = json.load(file)

print(data)

# Modifying JSON file
with open('files/data.json', 'r') as file:
    data = json.load(file)
    # New value
    data['profession'] = 'Just a noob programmer'
with open('files/data.json', 'w') as file:
    json.dump(data, file)


# Add id+1 to all proyects.
with open('files/data.json', 'r') as file:
    data = json.load(file)
    for i in data['proyects']:
        i['idp'] = i['idp']+1


with open('files/data.json', 'w') as file:
    json.dump(data, file)

# THE END.
