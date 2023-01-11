from typing import List, Optional
import torch
from pathlib import Path
from lxml import etree
from transformers import AutoTokenizer

from src.html_utils import get_cleaned_body

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

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

def generate_subtrees(root,m,s):
    pre_order = list(root.iter())
    post_order = list(postorder(root))
    return _get_subtrees(pre_order,post_order,m,s)

def _tokens_len(elements):
    if len(elements) == 0:
        return 0
    tokens = tokenize_elements(elements)
    return tokens["attention_mask"].sum().item()

# Appendix A: Algorithm 1
def _get_subtrees(pre_order,post_order, m,s):
    subtrees = []

    ### init first subtree
    new = []
    node_ids = {}
    for i,el in enumerate(pre_order):
        if _tokens_len(new) >= m:
            break
        new.append(el)  
        node_ids[el] = i

    while len(new) != 0:
        visited = [n for n,idx in node_ids.items() if idx < node_ids[new[0]] ]
        total_len = _tokens_len(visited + new)
        ###prune postorder
        for el in post_order:
            if el in new or total_len <= m:        
                break
            else:
                try:
                    visited.remove(el)
                    total_len -= _tokens_len([el])
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
                total_len -= _tokens_len([el_root])
                visited.pop(0)            
            else:
                el_last = new.pop()
                total_len -= _tokens_len([el_last])
        t = visited + new
        subtrees.append(t)

        el_last = new[-1]
        ### expand subtree
        new = []
        next_idx = node_ids[el_last] + 1
        for i,el in enumerate(pre_order[next_idx:],start=next_idx):
            if _tokens_len(new) >= s:
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
        "position_ids": [token_idx + i for i in range(len_tokens)], # p5
        **tokenizer_res
    }
    return result


# Local Subtree features
def get_tree_features(t):
    elem_idxs = {}
    result = {
            "node_ids": [] , # p0
            "parent_node_ids": [] , # p1
            "sibling_node_ids": [] , # p2
            "depth_ids": [] , # p3
            "tag_ids": [] , # p4
            "input_ids": [],
            "attention_mask" : []
        }
    reprs = []
    el_results = []
    for i,el in enumerate(t):
        el_parent = el.getparent()
        el_repr = represent_node(el)   
        reprs.append(el_repr)        
        
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
    # do batch tokenization
    tokenizer_res = tokenizer(reprs,return_tensors="pt",truncation=True,padding=True)     
    for el_result,input_ids,attn_mask in zip(el_results,tokenizer_res["input_ids"],tokenizer_res["attention_mask"]):
        len_tokens = attn_mask.sum().item()                
        input_ids = input_ids[:len_tokens].tolist()
        attn_mask = attn_mask[:len_tokens].tolist()
        for key in result:
            if key == "input_ids":
                result[key] += input_ids
            elif key == "attention_mask":
                result[key] += attn_mask
            else:
                result[key] += [el_result[key]] * len_tokens
    return result


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
        "input_ids": tokenizer.pad_token_id,
        "attention_mask": 0,
    }

    dom = get_cleaned_body(html_string)
    subtrees = generate_subtrees(dom,m,s) # requires tokenizer
    result = []
    for sub in subtrees:  
        data = get_tree_features(sub)     
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


    