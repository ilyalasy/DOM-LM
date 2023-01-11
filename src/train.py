
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

train_ds = dataset.SWDEDataset("/home/azureuser/dev/data/swde_preprocessed")
test_ds = dataset.SWDEDataset("/home/azureuser/dev/data/swde_preprocessed",split="test")

# tokenizer.pad_token = tokenizer.eos_token # why do we need this?
data_collator = DataCollatorForDOMNodeMask(tokenizer=tokenizer, mlm_probability=0.15)

#TODO: add evaluation metrics (ppl, etc.)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    num_train_epochs=5,
    warmup_ratio=0.1,
    learning_rate=1e-4
)

trainer = Trainer(
    model=domlm,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=data_collator,
)

trainer.train()
