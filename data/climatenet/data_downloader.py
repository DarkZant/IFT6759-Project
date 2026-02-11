import os
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://portal.nersc.gov/project/ClimateNet/climatenet_new/"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def download_split(split):
    data_dir = os.path.join(SCRIPT_DIR, split)

    os.makedirs(data_dir, exist_ok=True)
    print("Saving to:", data_dir)

    url = BASE_URL + split + "/"
    page = requests.get(url).text
    soup = BeautifulSoup(page, "html.parser")

    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.endswith(".nc"):
            file_url = url + href
            path = os.path.join(data_dir, href)
            print("Downloading", href)
            with requests.get(file_url, stream=True) as r:
                with open(path, "wb") as f:
                    for chunk in r.iter_content(8192):  # chunk of 8KB
                        f.write(chunk)


download_split("train")
download_split("test")
