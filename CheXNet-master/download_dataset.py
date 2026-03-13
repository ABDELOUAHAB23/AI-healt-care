import os
import urllib.request
import tarfile
import sys
from tqdm import tqdm

# Create a progress bar for downloads
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                           miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def main():
    # Create directories if they don't exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(base_dir, 'ChestX-ray14', 'images')
    os.makedirs(images_dir, exist_ok=True)

    print("Starting download of ChestX-ray14 dataset...")
    print("This will download approximately 45GB of data.")
    input("Press Enter to continue or Ctrl+C to cancel...")

    # URLs for the dataset parts
    base_url = "https://nihcc.box.com/shared/static/"
    links = [
        "vfk6huhje6ua88uk7gqkwqlv6r6vnm04.gz",
        "ospxu3vp0b3vshgq2wjg6neh3dpwu0g5.gz",
        "6uvkjdkbvf5bzzqxuk7x3sgy7dezr4cu.gz",
        "gxtz0j8uzvn06a65c6cvjh2kq2trgdle.gz",
        "p9qpktq6ek4tad8c5cpugh5qa5suuhb7.gz",
        "f1p9sjbfdr9v45w6vxel47z543jlh2q6.gz",
        "xcvd8c09f3x6r0k65j0h4x4yszkuvxzi.gz",
        "b7m11lfz48w7rsk0y32hlg9sla4a073c.gz",
        "9qm21owd6v5gmx44xhft6vl9ri29dwwz.gz",
        "c7m00e58s816rah6l9i6s8wqrsq4le5m.gz",
        "78m8jx6m3z02nn6xqfd7qtysxaoy91fi.gz",
        "kkg36petfyqmwk5kmx2v5ydt3jj1m8gr.gz"
    ]

    for i, link in enumerate(links, 1):
        url = base_url + link
        output_file = os.path.join(images_dir, f'images_{i}.tar.gz')
        
        print(f"\nDownloading part {i} of {len(links)}...")
        try:
            download_url(url, output_file)
            
            print(f"Extracting part {i}...")
            with tarfile.open(output_file) as tar:
                tar.extractall(path=images_dir)
            
            # Remove the tar.gz file after extraction
            os.remove(output_file)
            
        except Exception as e:
            print(f"Error downloading/extracting part {i}: {e}")
            continue

    print("\nDownload and extraction completed!")
    print("Please verify that all images are in the ChestX-ray14/images directory.")

if __name__ == "__main__":
    main()
