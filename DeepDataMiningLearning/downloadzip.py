import os
import zipfile

from pathlib import Path

import requests

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

if __name__ == "__main__":
    url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
    zipfilename = "pizza_steak_sushi.zip"
    downloadziponline(url, zipfilename)