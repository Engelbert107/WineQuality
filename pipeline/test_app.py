import requests

# The URL for your Flask app (where the model is deployed)
url = 'http://127.0.0.1:5000/predict_csv'

# Path to your unseen data CSV file
csv_file_path = '../data/unseen_wine_quality.csv'

# Make a POST request with the CSV file
with open(csv_file_path, 'rb') as f:
    response = requests.post(url, files={'file': f})

# Get the JSON response
response_json = response.json()

# Print the response with a descriptive message
if 'error' in response_json:
    print(f"Error: {response_json['error']}")
else:
    print("-----------"*10)
    print(f"Predicted values for wine quality : {response_json}")
    print("-----------"*10)
    print("Class 0 corresponds to wine quality ratings from 3 to 5, while Class 1 corresponds to wine ratings from 6 to 9.")
    print("-----------"*10)
    print()
