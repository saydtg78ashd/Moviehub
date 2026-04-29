from __future__ import annotations
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TARGET_DIR = DATA_DIR / "ml-latest-small"

def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with urlopen(URL) as response:
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(DATA_DIR)

if __name__ == "__main__":
    main()