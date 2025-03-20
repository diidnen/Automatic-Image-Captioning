import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import requests
import zipfile
import io as io_lib
from PIL import Image


# Create data directory if it doesn't exist
data_dir = 'data/coco'
os.makedirs(data_dir, exist_ok=True)

# Function to download COCO dataset
def download_coco_subset(data_dir, dataset='train2014'):
    """
    Download COCO dataset and annotations to specified directory
    """
    # Download annotations
    annotations_url = f'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
    print(f"Downloading annotations from {annotations_url}...")
    r = requests.get(annotations_url)#get request from the url
    z = zipfile.ZipFile(io_lib.BytesIO(r.content))
    z.extractall(data_dir)
    print(f"Annotations downloaded to {data_dir}/annotations/")
    
    # Download images (this can be large - train2014 is around 1GB)
    images_url = f'http://images.cocodataset.org/zips/{dataset}.zip'
    print(f"Downloading images from {images_url}...")
    print("This may take a while depending on your internet connection...")
    r = requests.get(images_url)#get request from the url
    z = zipfile.ZipFile(io_lib.BytesIO(r.content))
    z.extractall(data_dir)
    print(f"Images downloaded to {data_dir}/{dataset}/")

# Comment out the next line if you already have downloaded the dataset
download_coco_subset(data_dir)



