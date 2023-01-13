# DOM-LM
Unofficial implementation of [Dom-LM paper](https://arxiv.org/abs/2201.10608).

## Masked LM Training

1. Download [SWDE](http://web.archive.org/web/20210630013015/https://codeplexarchive.blob.core.windows.net/archive/projects/swde/swde.zip)
2. Run [preprocess_swde.py](src/preprocess_swde.py) for features extraction
3. Run either [train.py](src/train.py) or [train_mlm notebook](notebooks/train_mlm.ipynb)


## TODO
- [ ] Add Fine-tuning code for Attribute Extraction
- [ ] Add Fine-tuning code for Open Information Extraction
- [ ] Add Fine-tuning code for QA
- [ ] Train some models and add results
