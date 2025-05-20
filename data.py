from datasets import load_dataset
import typing
import torch
import json
from tokenizer import CharTokenizer

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L

from torch.nn.utils.rnn import pad_sequence


def load_bookcorpus_data(n: int = 1_000_000) -> typing.List[str]:
    """Load a subset of the BookCorpus dataset."""
    dataset = load_dataset('bookcorpus')  # This loads the train split by default
    data = dataset['train'][:n]  # Take first n entries
    return data['text']


def find_char_in_data(data: typing.List[str]) -> typing.List[str]:
    """Find and return all unique characters in the dataset."""
    characters = set()
    for sentence in data:
        characters.update(set(sentence))
    return sorted(list(characters))



class CharDataset(Dataset):

    def __init__(self, data: typing.List[str], tokenizer: CharTokenizer):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> torch.LongTensor:
        sentence = self.data[index]
        encoded = self.tokenizer.encode(sentence)
        return encoded


class CharDataModule(L.LightningDataModule):

    def __init__(self, data: typing.List[str], tokenizer: CharTokenizer, batch_size: int = 128):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        train_data, val_data, test_data = self.split(data)

        self.train_dataset = CharDataset(train_data, tokenizer)
        self.val_dataset = CharDataset(val_data, tokenizer)
        self.test_dataset = CharDataset(test_data, tokenizer)

    def split(self, data: typing.List[str]) -> typing.Tuple[typing.List[str], typing.List[str], typing.List[str]]:
        n_train = int(len(data) * 0.8)
        n_val = int(len(data) * 0.1)
        train_data = data[:n_train]
        val_data = data[n_train:n_train + n_val]
        test_data = data[n_train + n_val:]
        return train_data, val_data, test_data

    def collate_fn(self, samples: typing.List[torch.LongTensor]) -> torch.LongTensor:
        return pad_sequence(samples, batch_first=True, padding_value=self.tokenizer.pad_token)

    def common_dataloader(self, split: str) -> DataLoader:
        dataset = getattr(self, f'{split}_dataset')
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(split == 'train'),
            collate_fn=self.collate_fn
        )

    def train_dataloader(self) -> DataLoader:
        return self.common_dataloader('train')

    def val_dataloader(self) -> DataLoader:
        return self.common_dataloader('val')

    def test_dataloader(self) -> DataLoader:
        return self.common_dataloader('test')


if __name__ == "__main__":
    # Load and process the data
    data = load_corpus_dataset(10000)  # Limiting to 10k for testing
    characters = find_char_in_data(data)
    
    # Initialize tokenizer
    tokenizer = CharTokenizer(characters)
    tokenizer.save("token.json")

    # Create DataModule
    datamodule = CharDataModule(data, tokenizer, batch_size=128)

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Sample decoded text: {tokenizer.decode(datamodule.train_dataset[0])}")