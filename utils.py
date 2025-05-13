import os
import random

import numpy as np
import torch


class Collator_with_padding(object):
    def __init__(self, pad_id, device):
        self.pad_id = pad_id
        self.device = device

    def __call__(self, batch):
        seq_lens = [len(x) for x in batch]
        max_seq_len = max(seq_lens)

        new_batch = []
        for sequence in batch:
            for _ in range(max_seq_len - len(sequence)):
                sequence.append(self.pad_id)
            new_batch.append(sequence)

        sequences = torch.LongTensor(new_batch).to(self.device)

        new_batch = {"input_ids": sequences[:, :-1], "target_ids": sequences[:, 1:]}
        return new_batch


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    g = torch.Generator()
    g.manual_seed(seed)
