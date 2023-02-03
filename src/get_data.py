import requests
import sys
from pathlib import Path

from tqdm import tqdm

basedir = Path(__file__).parent.resolve()
url = "http://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz"
response = requests.get(url, stream=True)
total = int(response.headers.get("content-length", 0))

fname = "atlas-higgs-challenge-2014-v2.csv.gz"
fpath = Path(f"{basedir}/../data/{fname}")
if fpath.is_file():
    print("File already downloaded, skipping...")
    sys.exit(0)

with fpath.open("wb") as file, tqdm(
    desc=fname, total=total, unit="iB"
) as progress_bar:
    for data in response.iter_content(chunk_size=1024):
        size = file.write(data)
        progress_bar.update(size)
