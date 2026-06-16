import requests

payload = {
    "query": "Is there a fundamental right to equality of opportunity in public employment under the Indian Constitution?",
    "threshold": 0.70
}

response = requests.post("http://localhost:8080/api/legal/query", json=payload)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")
