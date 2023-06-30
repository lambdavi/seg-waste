import gdown
import subprocess
import os
url = 'https://drive.google.com/u/0/uc?id=14ThGc53okYC61AnTXFAofiYYY8PTZYtl&export=download'
output = 'dataset.zip'
gdown.download(url, output, quiet=False)
subprocess.call(["unzip", "dataset.zip"])
os.remove("dataset.zip")
