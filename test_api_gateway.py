import requests
 
url = 'https://exy970w5sd.execute-api.us-east-1.amazonaws.com/test/predict'
data = {'url': 'https://raw.githubusercontent.com/jdanussi/flowers-dataset/refs/heads/main/test/cape%20flower/image_03810.jpg'}
 
result = requests.post(url, json=data).json()
print(result)