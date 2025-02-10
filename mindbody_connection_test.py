import os
import http.client
from dotenv import load_dotenv

load_dotenv()

conn = http.client.HTTPSConnection("api.mindbodyonline.com")

API_KEY = os.getenv("MINDBODY_API_KEY")

headers = {
    'Api-Key': API_KEY,
    'SiteId': "-99",
    # 'authorization': "{staffUserToken}"
    }

conn.request("GET", "/public/v6/class/classes", headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))