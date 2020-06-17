import unittest

from neurolib.utils.collections import flatten_nested_dict


class TestCollections(unittest.TestCase):
    NESTED_DICT = {"a": {"b": "c", "d": "e"}}
    FLAT_DICT_DOT = {"a.b": "c", "a.d": "e"}

    def test_flatten_nested_dict(self):
        flat = flatten_nested_dict(self.NESTED_DICT, sep=".")
        self.assertDictEqual(flat, self.FLAT_DICT_DOT)


if __name__ == "__main__":
    unittest.main()
