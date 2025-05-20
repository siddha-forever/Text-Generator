# tokenizer.py
import json
import torch
from torch.nn.utils.rnn import pad_sequence

class CharTokenizer:
    def __init__(self, characters):
        self.characters = characters
        self.pad_token = 0
        self.bos_token = 1
        self.unk_token = 2
        self.vocab_size = len(characters) + 3

    def encode(self, sentence: str, add_bos_token: bool = False) -> torch.LongTensor:
        encoded = []
        if(add_bos_token):
            encoded.append(self.bos_token)
        for char in sentence:
            try:
                encoded.append(self.characters.index(char) + 3)
            except ValueError:
                encoded.append(self.unk_token)
        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, encoded):
        output = ''
        for idx in encoded:
            if idx < 3:
                continue
            output += self.characters[idx - 3]
        return output

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.characters, f)

    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            characters = json.load(f)
        return CharTokenizer(characters)