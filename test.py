import requests
 
url = "http://localhost:8080/2015-03-31/functions/function/invocations"
data = {"url": "https://raw.githubusercontent.com/jdanussi/flowers-dataset/refs/heads/main/test/cape%20flower/image_03810.jpg"}
 
result = requests.post(url, json=data).json()
print(result)