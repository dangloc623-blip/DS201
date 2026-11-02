import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

import json
import os

# gom các mẫu thành batch 
def collate_fn(items: list[dict]) -> dict:
    input_ids = [item["input_ids"] for item in items]
    lengths = torch.tensor([ids.shape[0] for ids in input_ids], dtype=torch.long)
    max_len = int(lengths.max())

    input_ids = torch.stack([
        F.pad(
            input, 
            pad=(0, max_len - input.shape[0]), 
            mode="constant", 
            value=0
        ) for input in input_ids
    ])

    labels = torch.stack([
        F.pad(
            item["label"], 
            pad=(0, max_len - item["label"].shape[0]), 
            mode="constant", 
            value=-100
        ) for item in items
    ])

    return {
        "input_ids": input_ids,     # (bs, max_len)
        "lengths": lengths,         # (bs, )
        "label": labels             # (bs, max_len)
    }

class Vocab:
    def __init__(self, train_path: str):
        self.pad = "<pad>"
        self.unk = "<unk>"

        all_words = set()
        labels = set()
        
        with open(train_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                for word in item["words"]:
                    all_words.add(word.lower())
                labels.update(item["tags"])

        # 0 = <pad>, 1 = <unk>, các từ khác bắt đầu từ 2
        self.w2i = {
            self.pad : 0,
            self.unk : 1
        }

        # word -> idx 
        for idx, word in enumerate(sorted(all_words), start=2):
            self.w2i[word] = idx

        # idx -> word 
        self.i2w = {idx: word for word, idx in self.w2i.items()}

        # label -> idx 
        self.l2i = {
            label: idx for idx, label in enumerate(sorted(labels))
        }

        # idx -> label
        self.i2l = {
            idx: label for label, idx in self.l2i.items()
        }

    @property
    def n_labels(self) -> int:
        return len(self.l2i)
    
    @property
    def len(self) -> int:
        return len(self.w2i)        # kích thước vocab -> số dòng của Embedding

    @property
    def pad_idx(self) -> int:
        return self.w2i[self.pad]

    # chuyển câu thành tensor số 
    def encode_sentence(self, words: list[str]) -> torch.Tensor:
        if len(words) == 0:
            words = [self.unk]

        unk_id = self.w2i[self.unk]
        word_ids = [self.w2i.get(word.lower(), unk_id) for word in words]

        return torch.tensor(word_ids, dtype=torch.long)

    # Chuyển label thành số  
    def encode_label(self, tags: list[str]) -> torch.Tensor:
        label_ids = [self.l2i[tag] for tag in tags]

        return torch.tensor(label_ids, dtype=torch.long)
    
    # chuyển idx -> label 
    def decode_label(self, label_ids: torch.Tensor) -> list:
        label_ids = label_ids.tolist()
        labels = [self.i2l[idx] for idx in label_ids]

        return labels

class PhoNER(Dataset):
    def __init__(self, path: str, vocab: Vocab):
        super().__init__()

        self._data = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self._data.append(json.loads(line))
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        item = self._data[idx]
        
        words = item["words"]
        tags = item["tags"]

        input_ids = self.vocab.encode_sentence(words)
        label_ids = self.vocab.encode_label(tags)

        return {
            "input_ids": input_ids,
            "label": label_ids
        }
