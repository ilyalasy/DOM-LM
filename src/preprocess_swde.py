import sys
from pathlib import Path
import tqdm
import pickle

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from src.preprocess import extract_features
from src.domlm import DOMLMConfig

SWDE_PATH = Path("/home/azureuser/dev/data/swde_html")
PROC_PATH = Path("/home/azureuser/dev/data/swde_preprocessed")
DOMAIN = "university"

config = DOMLMConfig.from_json_file("domlm-config/config.json")

files = sorted((SWDE_PATH / DOMAIN).glob("**/*.htm"))
pbar = tqdm.tqdm(files,total=len(files))
for path in pbar:
    pbar.set_description(f"Processing {path.relative_to(SWDE_PATH / DOMAIN)}")
    with open(path,'r') as f:
        html = f.read()
    features = extract_features(html,config)
    dir_name = PROC_PATH / DOMAIN / path.parent.name
    dir_name.mkdir(parents=True,exist_ok=True)
    with open(dir_name / path.with_suffix(".pkl").name,'wb') as f:
        pickle.dump(features,f)            

