import os

from torch.utils.data.dataloader import DataLoader
import pandas as pd
import json
from datasets import Dataset

def load_train_config(config_path="train_config.json"):
    """
    Load config for training.
    """
    with open(config_path, "r") as f:
        config = json.loads(f.read())
    return config

def load_dataset(ds, split, config):
    #fp = f"../data/{ds}/{split}.json"
    fp = os.path.join(config["data_json"],ds,f"{split}.json")
    #with open(fp, "r") as f:
    data = pd.read_json(fp, lines=True)
    data = data.to_dict('records')

    data = [dict(id=inst['id'],
            article=inst['article'],
            summary=" ".join(inst['lay_summary'])) for inst in data]
    return data

#def add_graph_text_data(graph_text_path, split, data):
#    """
#    Load augmented text data for the given dataset and split.
#    """
#    fp = f"{graph_text_path}/{split}_abstract_concepts_explanation.jsonl"
#    with open(fp, "r") as f:
#        explain_data = f.readlines()
#        explain_data = [json.loads(line) for line in explain_data]

#    for i, inst in enumerate(data):
#        aid = inst['id']
#        graph_explainations = explain_data[i]
#        assert aid == graph_explainations['id']
#        data[i]['article'] = "[GRAPH_FACTS]\n" + graph_explainations['text'] + "\n[ARTICLE]\n" + data[i]['article']

#    return data

def get_processed_elife_data(ds, tokenizer, config, split, shuffle=False):

    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        inputs = tokenizer(
            batch["article"],
            padding="max_length",
            truncation=True,
            max_length=config['encoder_max_length'],
        )
        outputs = tokenizer(
            batch["summary"],
            padding="max_length",
            truncation=True,
            max_length=config['decoder_max_length'],
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask

        # create 0 global_attention_mask lists
        batch["global_attention_mask"] = len(batch["input_ids"]) * [
            [0 for _ in range(len(batch["input_ids"][0]))]
        ]

        # since above lists are references, the following line changes the 0 index for all samples
        batch["global_attention_mask"][0][0] = 1
        batch["labels"] = outputs.input_ids

        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in batch["labels"]
        ]

        return batch


    data = load_dataset(ds, split, config)

    data = [{"idx": i, **x} for i, x in enumerate(data)]

    #if config['is_input_aug']:
    #    data = add_graph_text_data(config['graph_data_path'], "train", data)

    data = Dataset.from_list(data)

    # map train data
    data = data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=config['batch_size'],
    )

    # set Python list to PyTorch tensor
    data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels", "idx"],
    )

    dataloader = DataLoader(data, batch_size=config['batch_size'], shuffle=shuffle)

    return dataloader