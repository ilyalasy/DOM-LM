

label2id = {
    'name': 0,
    'phone': 1,
    'type': 2,
    'website': 3,
    'director': 4,
    'genre': 5,
    'mpaa_rating': 6,
    'title': 7
}

id2label = {
    0: 'name',
    1: 'phone',
    2: 'type',
    3: 'website',
    4: 'director',
    5: 'genre',
    6: 'mpaa_rating',
    7: 'title'
}

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
