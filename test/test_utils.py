import json
import unittest

from transformers import AutoTokenizer
from src.utils import get_processed_elife_data

class TestUtils(unittest.TestCase):
    def test_get_processed_elife_datase(self):
        ds = "elife"
        split = "val"

        with open("../src/train_config.json","r") as fd_config:
            config = json.load(fd_config)

        model = config["model_str"]

        tokenizer = AutoTokenizer.from_pretrained(model)
        datal = get_processed_elife_data(ds,tokenizer,config,split)
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
