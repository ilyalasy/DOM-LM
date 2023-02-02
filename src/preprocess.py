from typing import List, Optional, Any
from pathlib import Path
from lxml import etree
from transformers import AutoTokenizer

from src.html_utils import get_cleaned_body
from src.utils import truncate, label2id


tokenizer = AutoTokenizer.from_pretrained("roberta-base", )

NODE_ATTRIBUTES = ("class", "id", "title", "aria-label")

TAGS_PATH = Path(__file__).parents[1] / "data/tags.txt"
with open(TAGS_PATH, "r") as f:
    TAGS_OF_INTEREST = [l.strip() for l in f.readlines()]
TAG_TO_INT = dict((t, i) for i, t in enumerate(TAGS_OF_INTEREST))


tokens_cache = {}
def tokenize_elements(elements: List[etree._Element]):
    # TODO: add caching
    return tokenizer([represent_node(el) for el in elements], return_tensors="pt",truncation=True,padding=True)

def extract_element_info(
    element: etree._Element,
    all_text_content: bool = False,
    attrs_to_extract = NODE_ATTRIBUTES,
):
    inner_text = ""    
    if type(element) == etree._ElementUnicodeResult:
        if element.is_text or element.is_tail:
            inner_text = str(element)        
    else:
        if str(element.tag) == "img":
            inner_text = element.attrib.get("alt", "")
        else:
            if all_text_content:
                inner_text = element.text_content()
            else:
                inner_text = element.text if element.text else ""
            inner_text += element.tail if element.tail else ""

    attrs = {
        k: v for k, v in element.attrib.items() if v and k in attrs_to_extract
    }        
    return str(element.tag), inner_text.strip(), attrs

def represent_node(element: etree._Element):
    tag,text,attrs = extract_element_info(element)
    return f"{tag} {' '.join(attrs.values())} {text}".strip()


def postorder(root: etree._Element): 
    current_idx = 0
    stack = [] 
 
    while (root is not None or len(stack) != 0):
        if (root is not None):
            stack.append((root, current_idx))
            current_idx = 0
 
            if (len(root) >= 1):
                root = list(root)[0]
            else:
                root = None
            continue

        node,child_idx = stack.pop()
        yield node
 
        while (len(stack) != 0 and child_idx == len(stack[-1][0]) - 1):
            node,child_idx = stack[-1]
            stack.pop()
            yield node                     

        if (len(stack) != 0):
            root = list(stack[-1][0])[child_idx + 1]
            current_idx = child_idx + 1 
    return

def generate_subtrees(root: etree._Element, token_repr:dict, m, s):
    pre_order = list(root.iter())
    post_order = list(postorder(root))
    return _get_subtrees(pre_order, post_order, token_repr, m, s)

def _tokens_len(elements, token_repr):
    if len(elements) == 0:
        return 0
    return sum(sum(token_repr[el]['tokenizer_res']['attention_mask']) for el in elements)

# Appendix A: Algorithm 1
def _get_subtrees(pre_order, post_order, token_repr, m, s):
    subtrees = []

    ### init first subtree
    new = []
    node_ids = {}
    for i,el in enumerate(pre_order):
        if _tokens_len(new, token_repr) >= m:
            break
        new.append(el)  
        node_ids[el] = i

    while len(new) != 0:
        visited = [n for n,idx in node_ids.items() if idx < node_ids[new[0]] ]
        total_len = _tokens_len(visited + new, token_repr)
        ###prune postorder
        for el in post_order:
            if el in new or total_len <= m:        
                break
            else:
                try:
                    visited.remove(el)
                    total_len -= _tokens_len([el], token_repr)
                except ValueError:
                    pass
        ### prune root
        el_last = None
        while total_len > m:
            if len(visited) != 0:            
                el_root = visited[0]                
                num_child = sum(child in new or child in visited for child in el_root)        
            else:
                num_child = 2 # pop from new if there are no nodes in visited left

            if num_child < 2:
                total_len -= _tokens_len([el_root], token_repr)
                visited.pop(0)            
            else:
                el_last = new.pop()
                total_len -= _tokens_len([el_last], token_repr)
        t = visited + new
        subtrees.append(t)

        el_last = new[-1]
        ### expand subtree
        new = []
        next_idx = node_ids[el_last] + 1
        for i,el in enumerate(pre_order[next_idx:],start=next_idx):
            if _tokens_len(new, token_repr) >= s:
                break
            new.append(el)        
            node_ids[el] = i    
    return subtrees


def _get_depth(node: etree._Element,max_parent:Optional[etree._Element]=None):
    d = 0
    while node != max_parent:
        d += 1
        node = node.getparent()
    return d

# Global DOM per element feature
def get_features(element: etree._Element):    
    parent = element.getparent()    

    node_idx = -1
    parent_node_idx = -1
    prev_nodes = []
    for i,el in enumerate(element.getroottree().iter()):
        el_repr = represent_node(el)
        prev_nodes.append(el_repr)        
        if el == parent:
            parent_node_idx = i
        if el == element:
            node_idx = i
            break
    depth = _get_depth(element)
    node_idx_siblings = list(parent).index(element)
    tag_id = TAG_TO_INT.get(str(element.tag),TAG_TO_INT["UNK"])

    all_tokens = tokenizer(prev_nodes,return_tensors="pt",truncation=True,padding=True)
    token_idx = all_tokens['attention_mask'].sum().item() # total non-padding tokens

    element_repr = represent_node(element)
    tokenizer_res = tokenizer(element_repr,truncation=True)
    len_tokens = len(tokenizer_res["input_ids"])

    result = {
        "node_ids": [node_idx] * len_tokens, # p0
        "parent_node_ids": [parent_node_idx] * len_tokens, # p1
        "sibling_node_ids": [node_idx_siblings] * len_tokens, # p2
        "depth_ids": [depth] * len_tokens, # p3
        "tag_ids": [tag_id] * len_tokens, # p4
        # "position_ids": [token_idx + i for i in range(len_tokens)], # p5
        **tokenizer_res
    }
    return result


# Local Subtree features
def get_tree_features(t: etree._Element, token_repr: dict, max_seq_length:int):
    elem_idxs = {}
    result = {
            "node_ids": [] , # p0
            "parent_node_ids": [] , # p1
            "sibling_node_ids": [] , # p2
            "depth_ids": [] , # p3
            "tag_ids": [] , # p4
            # "position_ids": [], # p5
            "input_ids": [],
            "attention_mask" : []
        }
    reprs = []
    input_ids = []
    attention_mask = []
    el_results = []
    for i,el in enumerate(t):
        # el_repr = represent_node(el)   
        el_repr = token_repr[el]['el_repr']
        reprs.append(el_repr)        
        input_ids.append(token_repr[el]['tokenizer_res']['input_ids'])
        attention_mask.append(token_repr[el]['tokenizer_res']['attention_mask'])

        el_parent = el.getparent()
        parent_node_idx = 0 
        node_idx = i
        if el_parent in elem_idxs:        
            parent_node_idx = elem_idxs[el_parent] + 1 # Should it be different from node_id?
        elem_idxs[el] = node_idx

        siblings = [child for child in el_parent if child in elem_idxs]
        node_idx_siblings = siblings.index(el)
        
        depth = _get_depth(el,t[0])
        tag_id = TAG_TO_INT.get(str(el.tag),TAG_TO_INT["UNK"])

        el_results.append( {
            "node_ids": node_idx , # p0
            "parent_node_ids": parent_node_idx , # p1
            "sibling_node_ids": node_idx_siblings , # p2
            "depth_ids": depth, # p3
            "tag_ids": tag_id , # p4            
            }
        )
    max_length = max([len(x) for x in input_ids])
    if max_length > max_seq_length:
        max_length = max_seq_length
    input_ids = truncate(input_ids, max_length)
    attention_mask = truncate(attention_mask, max_length)
    for i, (el_result, input_ids, attn_mask) in enumerate(zip(el_results, input_ids, attention_mask)):
        len_tokens = sum(attn_mask)
        for key in result:
            if key == "input_ids":
                result[key] += input_ids
            elif key == "attention_mask":
                result[key] += attn_mask
            elif key == "position_ids":
                result[key] += [len(result[key])+j for j in range(len(input_ids))]
            else:
                result[key] += [el_result[key]] * len_tokens
    return result

def extract_token_for_nodes(dom: etree._Element) -> dict:
    '''Create dictionary that store el_reprs and tokenizer_res for each node in the DOM'''
    res = dict()
    for el in list(dom.iter()):
        el_repr = represent_node(el)
        tokenizer_res = tokenizer(el_repr,truncation=True)
        res[el] = {
            "el_repr": el_repr,
            "tokenizer_res": tokenizer_res
        }
    return res

def extract_features(html_string,config,m=None,s=128):
    if m is None:
        m = tokenizer.model_max_length
    padding_idxs = {
        "node_ids": config.node_pad_id,
        "parent_node_ids":config.node_pad_id,
        "sibling_node_ids": config.sibling_pad_id,
        "depth_ids": config.depth_pad_id,
        "tag_ids": config.tag_pad_id,
        # "tok_positions": max_position_embeddings + 1, # p5
        # "position_ids": config.max_position_embeddings,
        "input_ids": tokenizer.pad_token_id,
        "attention_mask": 0
    }

    dom = get_cleaned_body(html_string)
    token_repr = extract_token_for_nodes(dom)
    subtrees = generate_subtrees(dom, token_repr, m, s) # requires tokenizer
    result = []
    for sub in subtrees:  
        data = get_tree_features(sub, token_repr, m)
        # data = {}
        # for el in sub:
        #     feats = get_features(el) # should we consider all attributes locally?            
        #     new_data = {key: (data[key] if key in data else []) + feats[key] for key in feats if key != "input_ids"}
        #     new_data["input_ids"] = (data["input_ids"] if "input_ids" in data else []) + feats["input_ids"] 
        #     data = new_data        
        current_len = len(data["input_ids"])
        pad_len = max(m - current_len,0)
        for key in data:
            data[key] += [padding_idxs[key]] * pad_len
        result.append(data)
    return result

def get_tree_features_ae_task(t: List, token_repr: dict, node2label:dict, max_seq_length:int):
    elem_idxs = {}
    result = {
            "node_ids": [] , # p0
            "parent_node_ids": [] , # p1
            "sibling_node_ids": [] , # p2
            "depth_ids": [] , # p3
            "tag_ids": [] , # p4
            # "position_ids": [], # p5
            "input_ids": [],
            "attention_mask" : [],
            "labels": []
        }
    reprs = []
    input_ids = []
    attention_mask = []
    el_results = []
    labels = []
    for i,el in enumerate(t):
        el_repr = token_repr[el]['el_repr']
        reprs.append(el_repr)
        input_ids.append(token_repr[el]['tokenizer_res']['input_ids'])
        attention_mask.append(token_repr[el]['tokenizer_res']['attention_mask'])
        labels.append(align_labels(token_repr[el]['tokenizer_res'], el, node2label, label2id))
 
        el_parent = el.getparent()
        parent_node_idx = 0 
        node_idx = i
        if el_parent in elem_idxs:        
            parent_node_idx = elem_idxs[el_parent] + 1 # Should it be different from node_id?
        elem_idxs[el] = node_idx

        siblings = [child for child in el_parent if child in elem_idxs]
        node_idx_siblings = siblings.index(el)
        
        depth = _get_depth(el,t[0])
        tag_id = TAG_TO_INT.get(str(el.tag),TAG_TO_INT["UNK"])

        el_results.append( {
            "node_ids": node_idx , # p0
            "parent_node_ids": parent_node_idx , # p1
            "sibling_node_ids": node_idx_siblings , # p2
            "depth_ids": depth, # p3
            "tag_ids": tag_id , # p4 
            }
        )
    max_length = max([len(x) for x in input_ids])
    if max_length > max_seq_length:
        max_length = max_seq_length
    input_ids = truncate(input_ids, max_length)
    attention_mask = truncate(attention_mask, max_length)
    for i, (el_result, input_id, attn_mask, label) in enumerate(zip(el_results, input_ids, attention_mask, labels)):
        len_tokens = sum(attn_mask)
        for key in result:
            if key == "input_ids":
                result[key] += input_id
            elif key == "attention_mask":
                result[key] += attn_mask
            elif key == "labels":
                result[key] += label
            # elif key == "position_ids":
            #     result[key] += [len(result[key])+j for j in range(len(input_id))]
            else:
                result[key] += [el_result[key]] * len_tokens
    return result

def align_labels(tokenizer_res: Any, el: etree._Element, node2label:str, label2id:dict):
    label_ids = []
    word_ids = tokenizer_res.word_ids()
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        else:
            label_ids.append(label2id[node2label[el]] if el in node2label else 0)
    return label_ids

def assign_label_for_nodes(dom: etree._Element, text2label:dict):
    '''Create dictionary that store label for each nodes in the DOM'''
    res = dict()
    for el in list(dom.iter()):
        text = el.text.strip() if el.text else ""
        if text in text2label:
            res[el] = text2label[text]['label']
    return res

def extract_features_ae_task(html_string, text2label, config, m=None, s=128):
    if m is None:
        m = tokenizer.model_max_length
    padding_idxs = {
        "node_ids": config.node_pad_id,
        "parent_node_ids":config.node_pad_id,
        "sibling_node_ids": config.sibling_pad_id,
        "depth_ids": config.depth_pad_id,
        "tag_ids": config.tag_pad_id,
        # "position_ids": config.pad_token_id,
        "input_ids": tokenizer.pad_token_id,
        "attention_mask": 0,
        "labels": -100
    }

    dom = get_cleaned_body(html_string)
    token_repr = extract_token_for_nodes(dom)
    node2label = assign_label_for_nodes(dom, text2label)
    subtrees = generate_subtrees(dom, token_repr, m, s) # requires tokenizer
    result = []
    for sub in subtrees:
        data = get_tree_features_ae_task(sub, token_repr, node2label, m)
        current_len = len(data["input_ids"])
        pad_len = max(m - current_len,0)
        for key in data:
            data[key] += [padding_idxs[key]] * pad_len
        result.append(data)
    return result
