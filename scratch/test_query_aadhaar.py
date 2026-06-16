import requests

payload = {
    "query": "Who is eligible for an Aadhaar number under the Aadhaar Act?",
    "threshold": 0.70
}

response = requests.post("http://localhost:8080/api/legal/query", json=payload)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")
