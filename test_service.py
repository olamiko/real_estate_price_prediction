import requests

url = "http://127.0.0.1:8000/predict"
client = {"area": 3290, "bedrooms": 2, "bathrooms": 1, "stories": 1, "mainroad": "yes", "guestroom": "no", "basement": "no",
 "hotwaterheating": "yes", "airconditioning": "no", "parking": 1, "prefarea": "no", "furnishingstatus": "furnished"}

print(requests.post(url, json=client).json())