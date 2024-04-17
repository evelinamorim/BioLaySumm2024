import sys

import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler
import evaluate

from torch.optim import AdamW

from model import LEDKForConditionalGeneration
from model import GraphEncoder

import nltk

from utils import load_train_config, get_processed_elife_data

#config_path = sys.argv[1]
config_path = "train_config.json"

# Config
config = load_train_config(config_path)

# load tools
ds = "elife"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
accelerator = Accelerator()
tokenizer = AutoTokenizer.from_pretrained(config['model_str'])
metric = evaluate.load("rouge")

train_dataloader = get_processed_elife_data(ds, tokenizer, config, "val", shuffle=True)
val_dataloader = get_processed_elife_data(ds, tokenizer, config, "val", shuffle=False)
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

print("Loading model...")

# load model
model = LEDKForConditionalGeneration.from_pretrained(
    config['model_str'],
    use_cache=False,
    is_merge_encoders=config['is_merge_encoders'],
    is_graph_decoder=config['is_graph_decoder'],
    )
graph_encoder = GraphEncoder(config)

# set generate hyperparameters
model.config.num_beams = config['num_beams']
model.config.max_length = config['decoder_max_length']
model.config.min_length = config['min_length']
model.config.length_penalty = config['length_penalty']
model.config.no_repeat_ngram_size = config['no_repeat_ngram_size']

optimizer = AdamW(model.parameters(), lr=config['lr'])

train_dataloader, val_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, val_dataloader, model, optimizer
)

num_training_steps = config['num_epochs'] * len(train_dataloader)
lr_scheduler = get_scheduler(
  "linear",
  optimizer=optimizer,
  num_warmup_steps=0,
  num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))

epoch_measures = []

print("Training...")

completed_steps = 0

# Manual train loop
for epoch in range(config['num_epochs']):
    model.train()
    for step, batch in enumerate(train_dataloader):

        with accelerator.accumulate(model):
            # get graphs
            aids = batch["idx"]
            graph_enc_out = []
            for i, aid in enumerate(aids):
                graph_out = graph_encoder.forward(aid, "train", device)
                graph_out = torch.nn.functional.pad(graph_out,
                                                    (0, 0, 0, config['GAT_embedding_size'] - graph_out.shape[0]),
                                                    "constant", 0)
                graph_enc_out.append(graph_out)
            graph_enc_out = torch.stack(graph_enc_out, dim=1).to(torch.float16)
            graph_enc_out = graph_enc_out.view(len(aids), -1, config['GAT_embedding_size'])
            del batch['idx']
            batch['graph_encoder_outputs'] = graph_enc_out

            # get model outputs
            outputs = model(**batch)
            loss = outputs.loss
            # wandb.log({"loss": loss, "step": step})
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Check if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
