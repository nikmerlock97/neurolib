"""
Collections of custom data structures and types.
"""

from dpath.util import search, delete
from collections import MutableMapping

DEFAULT_STAR_SEPARATOR = "."


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    # now pickleable!!!
    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        self.update(state)


class star_dotdict(dotdict):
    """Support star notation in dotdict. Nested dicts are now treated as glob"""

    def __getitem__(self, attr):
        # if using star notaion -> return dict of all keys that match
        if "*" in attr:
            return search(self, attr, separator=DEFAULT_STAR_SEPARATOR)
        # otherwise -> basic dict.get
        else:
            return dict.get(self, attr)

    def __setitem__(self, attr, val):
        # if using star notaion -> search and set all keys matching
        if "*" in attr:
            for k, _ in search(self, attr, yielded=True, separator=DEFAULT_STAR_SEPARATOR):
                setattr(self, k, val)
        # otherwise -> just __setitem__
        else:
            dict.__setitem__(self, attr, val)

    def __delitem__(self, attr):
        # if using star notation -> use dpath's delete
        if "*" in attr:
            delete(self, attr, separator=DEFAULT_STAR_SEPARATOR)
        # otherwise -> just __delitem__
        else:
            dict.__delitem__(self, attr)


def flatten_nested_dict(nested_dict, parent_key="", sep=DEFAULT_STAR_SEPARATOR):
    """
    Return flat dictionary from nested one using `sep` as separator between
    levels of keys.

    :param nested_dict: nested dictionary to flatten
    :type nested_dict: dict
    :param parent_key: parent key for the new key in flat dictionary
    :type parent_key: str
    :param sep: separator between levels of keys
    :type sep: str
    """
    items = []
    for k, v in nested_dict.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
