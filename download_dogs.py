import os
import urllib.request
import tarfile

def progress_hook(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    print(f"Downloading: {percent}% [{count * block_size} / {total_size}]", end="\r", flush=True)

# Set the URL and filename for the dataset
urls = ["http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
         "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"]
filenames = ["stanford-dogs-dataset.tar", "annotation.tar"]
ROOT_DIR = '/root/bigfiles/datasets/'
dogs_dir = os.path.join(ROOT_DIR,"stanford-dogs-dataset")
if not os.path.exists(dogs_dir):
    os.makedirs(dogs_dir)
os.chdir(dogs_dir)

for url,filename in zip(urls,filenames):
    
    # Create a directory for the dataset and change to it

    # Download the dataset file
    urllib.request.urlretrieve(url, filename, reporthook=progress_hook)

    # Extract the contents of the tar file
    with tarfile.open(filename) as tar:
        tar.extractall()

    # Change back to the parent directory
    # os.chdir("..")
