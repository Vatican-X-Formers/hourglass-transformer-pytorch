from hourglass_transformer_pytorch import HourglassTransformerLM
from hourglass_transformer_pytorch.autoregressive_wrapper import \
    AutoregressiveWrapper

import random
import tqdm
import tensorflow_datasets as tfds
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 8
GRADIENT_ACCUMULATE_EVERY = 2
LEARNING_RATE = 2e-4
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 192


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
        dim=512,
        max_seq_len=SEQ_LEN,
        depth=(1, 2, 1),
        shorten_factor=3,
        heads=8,
        attn_resampling=False
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

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        if i % VALIDATE_EVERY == 0:
            print(f'training loss: {loss.item()}')
            model.eval()
            with torch.no_grad():
                loss = model(next(val_loader))
                print(f'validation loss: {loss.item()}')

        if i % GENERATE_EVERY == 0:
            model.eval()
            inp = random.choice(val_dataset)[:-1]
            prime = decode_tokens(inp)
            print(f'%s \n\n %s', (prime, '*' * 100))

            sample = model.generate(inp[None, ...], GENERATE_LENGTH)
            output_str = decode_tokens(sample[0])
            print(output_str)
