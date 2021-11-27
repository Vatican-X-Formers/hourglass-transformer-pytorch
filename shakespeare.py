from collections import defaultdict

from hourglass_transformer_pytorch import HourglassTransformerLM
from hourglass_transformer_pytorch.autoregressive_wrapper import \
    AutoregressiveWrapper

import os
import random
import tqdm
import tensorflow_datasets as tfds
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import neptune.new as neptune

# constants

NUM_BATCHES = 5000
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 3e-4
VALIDATE_EVERY = 100
GENERATE_EVERY = 1000
GENERATE_LENGTH = 192
SEQ_LEN = 192
use_rotary = True
sf_dropout = False

hierarchy = (1, 4, 1)
shorten_factor = 3
d_model = 512
n_heads = 8
attn_resampling = True

USE_NEPTUNE = True

if USE_NEPTUNE:
    run = neptune.init(
        project=os.environ['NEPTUNE_PROJECT'],
        api_token=os.environ['NEPTUNE_TOKEN'],
    )  # your credentials

params = {"bs": BATCH_SIZE,
          "grad_acc": GRADIENT_ACCUMULATE_EVERY,
          "lr": LEARNING_RATE,
          "seq_len": SEQ_LEN,
          "model_hierarchy": hierarchy,
          "shorten_factor": shorten_factor,
          "d_model": d_model,
          "n_heads": n_heads,
          "attn_resampling": attn_resampling,
          "use_rotary": use_rotary,
          "sf_dropout": sf_dropout}

if USE_NEPTUNE:
    run["parameters"] = params
# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data


def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))


if __name__ == '__main__':
    # instantiate GPT-like decoder model
    model = HourglassTransformerLM(
        num_tokens=256,
        dim=d_model,
        max_seq_len=SEQ_LEN,
        depth=hierarchy,
        shorten_factor=shorten_factor,
        heads=n_heads,
        attn_resampling=attn_resampling,
        use_rotary=use_rotary,
    )

    model = AutoregressiveWrapper(model)
    model.cuda()

    ds = tfds.load(name='tiny_shakespeare')

    def tfds_to_tensor(ds_split):
        example_bytes = next(iter(ds_split.take(1)))['text'].numpy()
        bytes_np = np.frombuffer(example_bytes, dtype=np.uint8)
        return torch.from_numpy(bytes_np)

    data_train = tfds_to_tensor(ds['train'])
    data_val = tfds_to_tensor(ds['validation'])


    # prepare enwik8 data
    class TextSamplerDataset(Dataset):
        def __init__(self, data, seq_len):
            super().__init__()
            self.data = data
            self.seq_len = seq_len

        def __getitem__(self, index):
            rand_start = torch.randint(0, self.data.size(0) - self.seq_len,
                                       (1,))
            full_seq = self.data[
                       rand_start: rand_start + self.seq_len + 1].long()
            return full_seq.cuda()

        def __len__(self):
            return self.data.size(0) // self.seq_len


    train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
    val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
    train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # optimizer

    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    min_sf, max_sf = (2, 2)
    min_eval_sf, max_eval_sf = (2, 2)
    # training
    train_losses = defaultdict(list)
    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
        model.train()

        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            sf = np.random.randint(min_sf, max_sf + 1)
            loss = model(next(train_loader), sf=sf)
            loss.backward()
            train_losses[sf].append(loss.item())

        if i % 50 == 0:
            for sf, losses in train_losses.items():
                avg_loss = np.array(losses).mean()
                print(f'training loss, sf={sf}: {avg_loss}')
                if USE_NEPTUNE:
                    run[f'train/loss_sf_{sf}'].log(step=i, value=avg_loss)
            train_losses = defaultdict(list)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        if i % VALIDATE_EVERY == 0:
            for sf in range(min_eval_sf, max_eval_sf+1):
                val_losses = []
                for val_batch in val_loader:
                    loss = model(val_batch, sf=sf)
                    val_losses.append(loss.item())
                val_loss = np.mean(np.array(val_losses))
                print(f'validation loss for sf={sf}: {val_loss}')
                if USE_NEPTUNE:
                    run[f'val/loss_sf_{sf}'].log(step=i, value=val_loss)

        if i % GENERATE_EVERY == 0:
            model.eval()
            inp = random.choice(val_dataset)[:-1]
            prime = decode_tokens(inp)
            print(f'%s \n\n %s', (prime, '*' * 100))

            sample = model.generate(inp[None, ...], GENERATE_LENGTH)
            output_str = decode_tokens(sample[0])
            print(output_str)
