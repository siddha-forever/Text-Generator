import pytorch_lightning as L
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from typing import Tuple, List

from tokenizer import CharTokenizer
from data import CharDataModule, load_bookcorpus_data


class Generator(L.LightningModule):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, tokenizer: CharTokenizer):
        self.save_hyperparameters(ignore=['tokenizer'])
        super().__init__()
        self.emb_layer = nn.Embedding(vocab_size, embedding_dim)
        self.rnn_layer = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.out_layer = nn.Linear(hidden_size, vocab_size)
        self.tokenizer = tokenizer

        self.bos_token = tokenizer.bos_token
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token)

    def forward(self, encoded: torch.LongTensor, hidden: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.emb_layer(encoded)
        rnn_out, hidden = self.rnn_layer(emb, hidden)
        out = self.out_layer(rnn_out)
        return out, hidden

    def prepend_bos(self, batch: torch.LongTensor) -> torch.LongTensor:
        bos = torch.full((batch.size(0), 1), fill_value=self.bos_token, device=batch.device)
        return torch.cat([bos, batch], dim=1)

    def training_step(self, batch: torch.LongTensor, batch_idx: int) -> torch.Tensor:
        inp = self.prepend_bos(batch)
        out, _ = self(inp)

        logits = out[:, :-1]
        targets = batch

        loss = self.loss_fn(logits.transpose(2, 1), targets)
        self.log('loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.LongTensor, batch_idx: int) -> torch.Tensor:
        inp = self.prepend_bos(batch)
        out, _ = self(inp)

        logits = out[:, :-1]
        targets = batch

        loss = self.loss_fn(logits.transpose(2, 1), targets)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch: torch.LongTensor, batch_idx: int) -> torch.Tensor:
        inp = self.prepend_bos(batch)
        out, _ = self(inp)

        logits = out[:, :-1]
        targets = batch

        loss = self.loss_fn(logits.transpose(2, 1), targets)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)

    def generate(self, prompt: str, n_token: int = 200) -> str:
        self.eval()
        encoded_prompt = self.tokenizer.encode(prompt)
        current_input = torch.tensor(encoded_prompt).unsqueeze(0).to(self.device).long()

        with torch.no_grad():
            out, hidden = self(current_input)
            generated_tokens = []
            current_input = current_input[:, -1:].long()

            for _ in range(n_token):
                out, hidden = self(current_input, hidden)
                logits = out.squeeze(1)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).long()
                generated_tokens.append(next_token.item())
                current_input = next_token

        return self.tokenizer.decode(torch.tensor(generated_tokens))


if __name__ == "__main__":
    # Load data
    data = load_bookcorpus_data(n=10_000)
    tokenizer = CharTokenizer.load('token.json')
    datamodule = CharDataModule(data, tokenizer)

    # Define model
    generator = Generator(tokenizer.vocab_size, 128, 512, tokenizer)

    # Define trainer
    trainer = L.Trainer(
        max_epochs=10,  # Changed from 1 to 10
        accelerator="auto",
        devices=1,
    )

    # Start training
    trainer.fit(model=generator, datamodule=datamodule)

    # Generate sample text
    prompt = 'i want to have'
    output = generator.generate(prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated Output: {output}")