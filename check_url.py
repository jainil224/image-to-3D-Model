import urllib.request
url = "https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/3dmodels/original/L/B01N2PLWIL.glb"
try:
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req) as response:
        print(f"Status: {response.status}")
        print(f"Headers: {response.headers}")
except Exception as e:
    print(f"Error: {e}")
