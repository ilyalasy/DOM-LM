import numpy as np


def padding(data, max_len, pad_value=1) -> list:
    """
    :param data: list of list
    :param max_len: int
    :param pad_value: int
    :return: list of list
    """
    pad_data = []
    for d in data:
        if len(d) < max_len:
            pad_data.append(d + [pad_value] * (max_len - len(d)))
        else:
            pad_data.append(d[:max_len])
    return pad_data

def truncate(data, max_len) -> list:
    """
    :param data: list of list
    :param max_len: int
    :return: list of list
    """
    trunc_data = []
    for d in data:
        if len(d) > max_len:
            trunc_data.append(d[:max_len])
        else:
            trunc_data.append(d)
    return trunc_data
