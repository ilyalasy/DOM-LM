import numpy as np
import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers import DataCollatorForWholeWordMask

def _numpy_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple)):
        examples = [np.array(e, dtype=np.int64) for e in examples]

    # Check if padding is necessary.
    length_of_first = len(examples[0])
    are_tensors_same_length = all(len(x) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return np.stack(examples, axis=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(len(x) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = np.full(shape=(len(examples), max_length), fill_value=tokenizer.pad_token_id, dtype=examples[0].dtype)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result

def _tf_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    import tensorflow as tf

    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple)):
        examples = [tf.convert_to_tensor(e, dtype=tf.int64) for e in examples]

    # Check if padding is necessary.
    length_of_first = len(examples[0])
    are_tensors_same_length = all(len(x) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return tf.stack(examples, axis=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(len(x) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    # result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    result = []
    rank = tf.rank(examples[0])
    paddings = np.zeros((rank, 2), dtype=np.int32)
    for example in examples:
        if tokenizer.padding_side == "right":
            paddings[0, 1] = max_length - len(example)
        else:
            paddings[0, 0] = max_length - len(example)
        result.append(tf.pad(example, paddings, constant_values=tokenizer.pad_token_id))
    return tf.stack(result, axis=0)


def tolist(x):
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):  # Checks for TF tensors without needing the import
        x = x.numpy()
    return x.tolist()


@dataclass
class DataCollatorForDOMNodeMask(DataCollatorForWholeWordMask):
    """
    Data collator used for language modeling that masks entire DOM nodes.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling

    """

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            input_ids = [e["input_ids"] for e in examples]
            node_ids = [e["node_ids"] for e in examples]
        else:
            raise ValueError(
                "Examples required to have both input_ids and node_ids"                
            )

        batch_input = _torch_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        mask_labels = []
        for node_id in node_ids:            
            mask_labels.append(self._whole_node_mask(node_id))
        batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        inputs, labels = self.torch_mask_tokens(batch_input, batch_mask)
        return {"input_ids": inputs, "labels": labels}

    def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            input_ids = [e["input_ids"] for e in examples]
            node_ids = [e["node_ids"] for e in examples]
        else:
            raise ValueError(
                "Examples required to have both input_ids and node_ids"                
            )

        batch_input = _tf_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        mask_labels = []
        for node_id in node_ids:            
            mask_labels.append(self._whole_node_mask(node_id))
        batch_mask = _tf_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        inputs, labels = self.tf_mask_tokens(batch_input, batch_mask)
        return {"input_ids": inputs, "labels": labels}

    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(examples[0], Mapping):
            input_ids = [e["input_ids"] for e in examples]
            node_ids = [e["node_ids"] for e in examples]
        else:
            raise ValueError(
                "Examples required to have both input_ids and node_ids"                
            )

        batch_input = _numpy_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        mask_labels = []
        for node_id in node_ids:            
            mask_labels.append(self._whole_node_mask(node_id))
        batch_mask = _numpy_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        inputs, labels = self.numpy_mask_tokens(batch_input, batch_mask)
        return {"input_ids": inputs, "labels": labels}

    def _whole_node_mask(self, input_nodes: List[int], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole node mask proxy
        """
        # selecting random nodes
        unique_nodes = list(set(input_nodes))
        unique_nodes.remove(max(unique_nodes)) # remove pad node
        random.shuffle(unique_nodes)

        num_to_predict = min(max_predictions, max(1, int(round(len(input_nodes) * self.mlm_probability))))
        masked_idxs = []        
        for unique_node in unique_nodes:
            if len(masked_idxs) >= num_to_predict:
                break

            indexes = [i for i,n in enumerate(input_nodes) if n == unique_node]

            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_idxs) + len(indexes) > num_to_predict:
                continue
            
            masked_idxs.extend(indexes)

        mask_labels = [1 if i in masked_idxs else 0 for i in range(len(input_nodes))]
        return mask_labels