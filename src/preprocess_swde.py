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

SWDE_PATH = Path("/home/c4i/crawler/smart_crawler_research/DOM-LM/data/swde_html/sourceCode/sourceCode")
PROC_PATH = Path("/home/c4i/crawler/smart_crawler_research/DOM-LM/data/swde_preprocessed")
DOMAIN = "university"

config = DOMLMConfig.from_json_file("domlm-config/config.json")

start_from = 0
files = sorted((SWDE_PATH / DOMAIN).glob("**/*.htm"))[start_from:]
pbar = tqdm.tqdm(files,total=len(files))
errors = []
for path in pbar:    
    pbar.set_description(f"Processing {path.relative_to(SWDE_PATH / DOMAIN)}")
    with open(path,'r') as f:
        html = f.read()
    try:
        features = extract_features(html,config)
        dir_name = PROC_PATH / DOMAIN / path.parent.name
        dir_name.mkdir(parents=True,exist_ok=True)
        with open(dir_name / path.with_suffix(".pkl").name,'wb') as f:
            pickle.dump(features,f)          
    except Exception as e:
        print(e)
        errors.append(path)
        pass
print(f"Total errors: {len(errors)}")


