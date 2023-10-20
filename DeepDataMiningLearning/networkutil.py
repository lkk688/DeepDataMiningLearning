import os
from pathlib import Path
from zipfile import ZipFile
#proxy = 'http://<user>:<pass>@<proxy>:<port>'
os.environ['http_proxy'] = 'http://172.16.1.2:3128'
os.environ['HTTP_PROXY'] = 'http://172.16.1.2:3128'
os.environ['https_proxy'] = 'https://172.16.1.2:3128'
os.environ['HTTPS_PROXY'] = 'https://172.16.1.2:3128'
import zipfile

from pathlib import Path

import requests

def _get_redirect_url(url: str, max_hops: int = 3) -> str:
    USER_AGENT = "pytorch/vision"
    initial_url = url
    headers = {"Method": "HEAD", "User-Agent": USER_AGENT}
    for _ in range(max_hops + 1):
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
            if response.url == url or response.url is None:
                return url

            url = response.url
    else:
        raise RecursionError(
            f"Request to {initial_url} exceeded {max_hops} redirects. The last redirect points to {url}."
        )

def download(url, dir='.', filename='tmp.zip', unzip=True, delete=False, curl=True):
    #f = Path("/home/010796032/MyRepo/DeepDataMiningLearning/data") / "coco128_with_yaml.zip"  # filename
    dir=Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    f = dir / filename
    if Path(url).is_file():  # exists in current path
        #Path(url).rename(f)  # move to dir
        print("File already exist")
    elif not f.exists():
        print(f'Downloading {url} to {f}...')
        if curl:
            os.system(f"curl -L '{url}' -o '{f}' --retry 9 -C -")  # curl download, retry and resume on fail
        else:
            torch.hub.download_url_to_file(url, f, progress=True)  # torch download
    if unzip and f.suffix in ('.zip', '.gz'):
        print(f'Unzipping {f}...')
        if f.suffix == '.zip':
            ZipFile(f).extractall(path=dir)  # unzip
        elif f.suffix == '.gz':
            os.system(f'tar xfz {f} --directory {f.parent}')  # unzip
        if delete:
            f.unlink()  # remove zip

def downloadurls(dir, urls):
    #dir = Path("/home/010796032/MyRepo/DeepDataMiningLearning/data") #root path
    dir = Path(dir)
    for u in [urls] if isinstance(urls, (str, Path)) else urls:
        filename = Path(u).name
        newurl = _get_redirect_url(u, max_hops=3)
        download(newurl, dir, filename, unzip=True, delete=False, curl=True)


def downloadziponline(url, zipfilename, download_folder='../data/', image_path='downloadedimage'):
    # Setup path to data folder
    data_path = Path(download_folder)
    image_path = data_path / image_path

    print("Current folder:", os.getcwd())

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Download
        with open(data_path / zipfilename, "wb") as f:
            request = requests.get(url)
            print("Downloading zip data...")
            f.write(request.content)
        print("Finish downloading")
        # Unzip 
        with zipfile.ZipFile(data_path / zipfilename, "r") as zip_ref:
            print("Unzipping pizza, steak, sushi data...") 
            zip_ref.extractall(image_path)
        print("Unzip success")
        # Remove .zip file
        #os.remove(data_path / zipfilename)


import torch
import torchvision
import torchvision.datasets as datasets
import os
import urllib.request
with urllib.request.urlopen('http://python.org/') as response:
   html = response.read()
   print(html)


if __name__ == "__main__":
    url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
    zipfilename = "pizza_steak_sushi.zip"
    downloadziponline(url, zipfilename)
    
#if __name__ == '__main__':
#os.system(f"curl -L '{url}' -o '{f}' --retry 9 -C -")  # curl download, retry and resume on fail
#torch.hub.download_url_to_file(url, f, progress=True)  # torch download