import argparse

import sys
from pathlib import Path
import tqdm
import pickle

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from src.preprocess import extract_features, extract_features_ae_task
from src.domlm import DOMLMConfig


def extract_labels(label_files):
    label_info = {}
    for file in label_files:
        label = file.name.split('-')[-1].replace('.txt', '')
        with open(file, 'r') as f:
            content = f.readlines()
            for line in content[2:]:
                page_id = line.split('\t')[0]
                if page_id not in label_info:
                    label_info[page_id] = {}
                nums = line.split('\t')[1]
                value = line.split('\t')[2].strip()
                label_info[page_id][label] = {
                    'nums': nums,
                    'value': value,
                }
    return label_info
    
def preprocess_swde(input_dir, config, output_dir, domains):
    SWDE_PATH = Path(input_dir)
    PROC_PATH = Path(output_dir)
    DOMAINS = domains

    config = DOMLMConfig.from_json_file(config)

    start_from = 0
    for domain in DOMAINS:
        files = sorted((SWDE_PATH / domain).glob("**/*.htm"))[start_from:]
        pbar = tqdm.tqdm(files,total=len(files))
        errors = []
        for path in pbar:    
            pbar.set_description(f"Processing {path.relative_to(SWDE_PATH / domain)}")
            with open(path,'r') as f:
                html = f.read()
            try:
                features = extract_features(html,config)
                dir_name = PROC_PATH / domain / path.parent.name
                dir_name.mkdir(parents=True,exist_ok=True)
                with open(dir_name / path.with_suffix(".pkl").name,'wb') as f:
                    pickle.dump(features,f)          
            except Exception as e:
                print(e)
                errors.append(path)
                pass
        print(f"Total errors: {len(errors)}")

def preprocess_swde_attribute_extraction(input_dir, config_file, output_dir, domains):
    SWDE_PATH = Path(input_dir)
    LABEL_PATH = SWDE_PATH / 'groundtruth'
    PROC_PATH = Path(output_dir)
    DOMAINS = domains

    config = DOMLMConfig.from_json_file(config_file)

    for domain in DOMAINS:
        for website_dir in (SWDE_PATH / domain).iterdir():
            if not website_dir.is_dir():
                continue
            files = sorted((website_dir.glob("./*.htm")))
            website_name = website_dir.name.split('-')[1][:website_dir.name.split('-')[1].index('(')]
            label_files = sorted((LABEL_PATH / domain).glob(f'{domain}-{website_name}*'))
            label_infos = extract_labels(label_files)
            pbar = tqdm.tqdm(files, total=len(files))
            errors = []
            for path in pbar:
                pbar.set_description(f"Processing {path.relative_to(SWDE_PATH / domain)}")
                with open(path,'r') as f:
                    html = f.read()
                try:
                    label2text = label_infos[path.name.split('.')[0]]
                    text2label = {v['value']: {'label':k, 'nums':v['nums']} for k,v in label2text.items()}
                    features = extract_features_ae_task(html, text2label, config)
                    dir_name = PROC_PATH / domain / path.parent.name
                    dir_name.mkdir(parents=True,exist_ok=True)
                    with open(dir_name / path.with_suffix(".pkl").name,'wb') as f:
                        pickle.dump(features, f)
                except Exception as e:
                    print(e)
                    errors.append(path)
                    pass
        print(f"Total errors: {len(errors)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='attribute_extraction', help='preprocess data for tasks', choices=['domlm', 'attribute_extraction'])
    parser.add_argument('--input_dir', type=str, default='data/swde_html/sourceCode/sourceCode', help='data directory')
    parser.add_argument('--config', type=str, default='domlm-config/config.json', help='config file')
    parser.add_argument('--output_dir', type=str, default='data/swde_ae_preprocessed', help='output directory')
    parser.add_argument('--domains', type=str, default='auto,book,camera,job,movie,nbaplayer,restaurant,university', help='domains')
    args = parser.parse_args()

    task = args.task
    input_dir = args.input_dir
    config = args.config
    output_dir = args.output_dir
    domains = args.domains.split(',')

    if task == 'domlm':
        preprocess_swde(input_dir, config, output_dir, domains)
    elif task == 'attribute_extraction':
        preprocess_swde_attribute_extraction(input_dir, config, output_dir, domains)
