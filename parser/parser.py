import requests
from bs4 import BeautifulSoup

url = "https://marvel.fandom.com/wiki/Beyonder_(Earth-616)"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")
cleaned = soup.text.replace("\n\n", "")
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(cleaned)
