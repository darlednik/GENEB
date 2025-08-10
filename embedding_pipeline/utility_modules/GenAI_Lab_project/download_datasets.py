import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

import requests
from tqdm import tqdm
import gzip
import shutil

def download_datasets():
    def download_genomes(url, dest_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
        # Download the genomes
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kilobyte
    
        tqdm_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(dest_path))
        
        with open(dest_path, 'wb') as f:
            for data in response.iter_content(block_size):
                tqdm_bar.update(len(data))
                f.write(data)
        tqdm_bar.close()
    
        if total_size != 0 and tqdm_bar.n != total_size:
            print("WARNING: Downloaded size does not match expected size.")
    
    def download_and_unzip(url, output_path):
        gz_path = output_path + ".gz"
        
        download_genomes(url, gz_path)
    
        print(f"Unzipping {gz_path} ...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        os.remove(gz_path)
        print(f"Saved to {output_path}")
    
    # Set download directory
    download_dir = "data/genomes"
    os.makedirs(download_dir, exist_ok=True)
    
    # Download and unzip genomes
    download_and_unzip(
        "http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz", 
        os.path.join(download_dir, "hg19.fa")
    )

    download_and_unzip(
        "http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz", 
        os.path.join(download_dir, "hg38.fa")
    )
    
    download_and_unzip(
        "http://hgdownload.cse.ucsc.edu/goldenPath/galGal4/bigZips/galGal4.fa.gz", 
        os.path.join(download_dir, "galGal4.fa")
    )
    
    download_and_unzip(
        "http://hgdownload.cse.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz", 
        os.path.join(download_dir, "mm10.fa")
    )

download_datasets()
