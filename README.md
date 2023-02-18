# DOM-LM - Pytorch (wip)
Unofficial implementation of [Dom-LM paper](https://arxiv.org/abs/2201.10608).

## Masked LM Training

1. Download [SWDE](http://web.archive.org/web/20210630013015/https://codeplexarchive.blob.core.windows.net/archive/projects/swde/swde.zip)
   - Or: Download from torrent [SWDE](https://academictorrents.com/details/411576c7e80787e4b40452360f5f24acba9b5159)
3. Run [preprocess_swde.py](src/preprocess_swde.py) for features extraction
4. Run either [train.py](src/train.py) or [train_mlm notebook](notebooks/train_mlm.ipynb)


## TODO
- [ ] Add Fine-tuning code for Attribute Extraction
- [ ] Add Fine-tuning code for Open Information Extraction
- [ ] Add Fine-tuning code for QA
- [ ] Train some models and compare results
