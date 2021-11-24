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

NUM_BATCHES = int(1e4)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 3e-4
VALIDATE_EVERY = 100
GENERATE_EVERY = 1000
GENERATE_LENGTH = 192
SEQ_LEN = 192
VAL_STEPS = 30
sf_dropout = False

hierarchy = (1, 2, 1)
shorten_factor = 3
d_model = 512
n_heads = 8
attn_resampling = False

USE_NEPTUNE = False

if USE_NEPTUNE:
    run = neptune.init(
        project=os.environ['NEPTUNE_PROJECT'],
        api_token=os.environ['NEPTUNE_TOKEN'],
    )  # your credentials

params = {"bs": BATCH_SIZE,
          "grad_acc": GRADIENT_ACCUMULATE_EVERY,
          "val_steps": VAL_STEPS,
          "lr": LEARNING_RATE,
          "seq_len": SEQ_LEN,
          "model_hierarchy": hierarchy,
          "shorten_factor": shorten_factor,
          "d_model": d_model,
          "n_heads": n_heads,
          "attn_resampling": attn_resampling,
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
    val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))

    # optimizer

    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training

    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
        model.train()

        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            loss = model(next(train_loader))
            loss.backward()

        if i % 50 == 0:
            print(f'training loss: {loss.item()}')
            if USE_NEPTUNE:
                run['train/loss'].log(step=i, value=loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        if i % VALIDATE_EVERY == 0:
            val_losses = []
            for _ in range(VAL_STEPS):
                loss = model(next(val_loader))
                val_losses.append(loss.item())
            val_loss = np.mean(np.array(val_losses))
            print(f'validation loss: {val_loss}')
            if USE_NEPTUNE:
                run['val/loss'].log(step=i, value=val_loss)

        if i % GENERATE_EVERY == 0:
            model.eval()
            inp = random.choice(val_dataset)[:-1]
            prime = decode_tokens(inp)
            print(f'%s \n\n %s', (prime, '*' * 100))

            sample = model.generate(inp[None, ...], GENERATE_LENGTH)
            output_str = decode_tokens(sample[0])
            print(output_str)
