
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from collections import OrderedDict

import src.domlm as model 
import src.dataset as dataset
from src.data_collator import DataCollatorForDOMNodeMask



tokenizer = AutoTokenizer.from_pretrained("roberta-base")
roberta = AutoModel.from_pretrained("roberta-base")
roberta_config = roberta.config


roberta_config_dict = roberta_config.to_dict()
roberta_config_dict["_name_or_path"] = "domlm"
roberta_config_dict["architectures"] = ["DOMLMForMaskedLM"]
domlm_config = model.DOMLMConfig.from_dict(roberta_config_dict)
# domlm_config.save_pretrained("../domlm-config/")
domlm = model.DOMLMForMaskedLM(domlm_config)

state_dict = OrderedDict((f"domlm.{k}",v) for k,v in roberta.state_dict().items())
domlm.load_state_dict(state_dict,strict=False)

dataset_path = ROOT / "data/swde_preprocessed"
print(f"Loading datasets from {dataset_path}...")
train_ds = dataset.SWDEDataset(dataset_path)
test_ds = dataset.SWDEDataset(dataset_path,split="test")

# tokenizer.pad_token = tokenizer.eos_token # why do we need this?
data_collator = DataCollatorForDOMNodeMask(tokenizer=tokenizer, mlm_probability=0.15)

# install apex:
# comment lines 32-40 in apex/setup.py
# pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

#TODO: add evaluation metrics (ppl, etc.)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    # optim="adamw_apex_fused", # only with apex installed
    weight_decay=0.01,
    num_train_epochs=5,
    warmup_ratio=0.1,
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=4,
    # gradient_checkpointing=True, # vram is enough without checkpointing on A4000
    bf16 = True, # If not Ampere: fp16 = True
    tf32 = True, # Ampere Only
    dataloader_num_workers=8,
    dataloader_pin_memory=True
)

trainer = Trainer(
    model=domlm,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=data_collator,
)

trainer.train()
