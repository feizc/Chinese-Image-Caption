from PIL import Image
import requests 

url = 'http://download-image.sankuai.com/ugcpic/d755e381339f49c126f0c5b91c67da78'
Image.open(requests.get(url, stream=True).raw) 
