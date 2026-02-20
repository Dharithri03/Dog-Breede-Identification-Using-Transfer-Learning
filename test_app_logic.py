
import requests

url = 'http://127.0.0.1:5000/predict'
files = {'file': open('test_dog.jpg', 'rb')}

try:
    response = requests.post(url, files=files)
    print("Status Code:", response.status_code)
    if response.status_code == 200:
        print("Prediction Successful!")
        print("Response contains 'Identified Breed' text:", 'Identified Breed' in response.text)
        # simplistic check for HTML content
    else:
        print("Failed:", response.text)
except Exception as e:
    print("Connection Error:", e)
