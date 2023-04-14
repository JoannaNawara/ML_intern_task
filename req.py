import requests
import json

# Define the URL endpoint and input data
url = "http://127.0.0.1:5000/predict?model=dct"
data = {"Elevation": 2596,
 'Aspect': 51,
 'Slope': 3,
 'Horizontal_Distance_To_Hydrology': 258,
 'Vertical_Distance_To_Hydrology': 0,
 'Horizontal_Distance_To_Roadways': 510,
 'Hillshade_9am': 221,
 'Hillshade_Noon': 232,
 'Hillshade_3pm': 148,
 'Horizontal_Distance_To_Fire_Points': 6279,
 'Wilderness_Area1': 1,
 'Wilderness_Area2': 0,
 'Wilderness_Area3': 0,
 'Wilderness_Area4': 0,
 'Soil_Type1': 0,
 'Soil_Type2': 0,
 'Soil_Type3': 0,
 'Soil_Type4': 0,
 'Soil_Type5': 0,
 'Soil_Type6': 0,
 'Soil_Type7': 0,
 'Soil_Type8': 0,
 'Soil_Type9': 0,
 'Soil_Type10': 0,
 'Soil_Type11': 0,
 'Soil_Type12': 0,
 'Soil_Type13': 0,
 'Soil_Type14': 0,
 'Soil_Type15': 0,
 'Soil_Type16': 0,
 'Soil_Type17': 0,
 'Soil_Type18': 0,
 'Soil_Type19': 0,
 'Soil_Type20': 0,
 'Soil_Type21': 0,
 'Soil_Type22': 0,
 'Soil_Type23': 0,
 'Soil_Type24': 0,
 'Soil_Type25': 0,
 'Soil_Type26': 0,
 'Soil_Type27': 0,
 'Soil_Type28': 0,
 'Soil_Type29': 1,
 'Soil_Type30': 0,
 'Soil_Type31': 0,
 'Soil_Type32': 0,
 'Soil_Type33': 0,
 'Soil_Type34': 0,
 'Soil_Type35': 0,
 'Soil_Type36': 0,
 'Soil_Type37': 0,
 'Soil_Type38': 0,
 'Soil_Type39': 0,
 'Soil_Type40': 0}

# Convert the input data to a JSON payload
payload = json.dumps(data)

# Set the headers and send the request
headers = {'Content-Type': 'application/json'}
response = requests.post(url, headers=headers,  data=payload)

# Print the response
print(response.json())
