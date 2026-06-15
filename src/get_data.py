import kagglehub
import shutil
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

source_file = Path(path) / "IMDB Dataset.csv"
target_file = DATA_DIR / "IMDB Dataset.csv"

shutil.copy(source_file, target_file)

print(f"Dataset saved to {target_file}")