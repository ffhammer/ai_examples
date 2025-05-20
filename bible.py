import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from src.layers import SelfAttentionBlock, PositionalEncoding, Vocab
import random

random.seed(0)


class SmallLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_tokens: int,
        dim_size: int = 128,
        n_blocks: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embed = Vocab(dim_size, vocab_size)
        self.pos_enc = PositionalEncoding(dim_size, dropout, 256)
        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(dim_size, num_heads, dropout) for _ in range(n_blocks)]
        )
        self.ln = nn.LayerNorm(dim_size)
        self.head = nn.Linear(dim_size, vocab_size)

    def forward(self, x: torch.LongTensor, mask: torch.BoolTensor) -> torch.Tensor:
        tok = self.token_embed(x)  # (B, T, D)
        h = self.pos_enc(tok.transpose(0, 1)).transpose(0, 1)  # (B, T, D)
        for blk in self.blocks:
            h = blk(h, mask)
        return self.head(self.ln(h))  # (B, T, V)


class LLMDataset(Dataset):
    def __init__(
        self, texts: list[str], tokenizer: AutoTokenizer, seq_len: int
    ) -> None:
        enc = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=seq_len + 1,
        )
        ids = enc["input_ids"]
        self.x = ids[:, :-1]
        self.y = ids[:, 1:]

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class BibleDataModule(pl.LightningDataModule):
    def __init__(
        self, tokenizer: AutoTokenizer, batch_size: int = 16, seq_len: int = 64
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len

    def setup(self, stage=None):
        ds = load_dataset("bible_para", "de-en")

        texts = [t["translation"]["de"] for t in ds["train"] if t]
        random.shuffle(texts)
        train = texts[: int(len(texts) * 0.95)]
        val = texts[int(len(texts) * 0.95) :]

        self.train_ds = LLMDataset(train, self.tokenizer, self.seq_len)
        self.val_ds = LLMDataset(val, self.tokenizer, self.seq_len)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)


class BibleLLM(pl.LightningModule):
    def __init__(self, tokenizer: AutoTokenizer, **hparams) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.tokenizer = tokenizer
        self.model = SmallLLM(
            vocab_size=tokenizer.vocab_size,
            max_tokens=hparams["seq_len"],
            dim_size=hparams["dim_size"],
            n_blocks=hparams["n_blocks"],
            num_heads=hparams["num_heads"],
            dropout=hparams["dropout"],
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.LongTensor, mask: torch.BoolTensor) -> torch.Tensor:
        return self.model(x, mask)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pad_mask = x != self.tokenizer.pad_token_id
        causal = torch.tril(
            torch.ones(x.size(1), x.size(1), device=self.device, dtype=torch.bool)
        )
        mask = pad_mask.unsqueeze(1) & causal.unsqueeze(0)
        logits = self(x, mask)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        pad_mask = x != self.tokenizer.pad_token_id
        causal = torch.tril(
            torch.ones(x.size(1), x.size(1), device=self.device, dtype=torch.bool)
        )
        mask = pad_mask.unsqueeze(1) & causal.unsqueeze(0)
        logits = self(x, mask)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log("val_loss", loss, prog_bar=True)

        if batch_idx == 0:
            prompt = "Und Gott sprach"
            out = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=x.size(1),
                truncation=True,
            )["input_ids"].to(self.device)
            out = out[:, : x.size(1)]  # Ensure correct shape (1, T)
            for _ in range(20):
                L2 = out.size(1)
                pad2 = out != self.tokenizer.pad_token_id
                causal2 = torch.tril(
                    torch.ones(L2, L2, device=self.device, dtype=torch.bool)
                )
                mask2 = pad2.unsqueeze(1) & causal2.unsqueeze(0)
                logits2 = self(out, mask2)
                next_id = logits2[:, -1].argmax(-1, keepdim=True)
                if next_id.item() == self.tokenizer.eos_token_id:
                    break
                out = torch.cat([out, next_id], dim=1)
            text = self.tokenizer.decode(
                out[0].cpu().tolist(), skip_special_tokens=True
            )
            print(f"\nSampled text: {text}")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    dm = BibleDataModule(tokenizer)
    model = BibleLLM(
        tokenizer,
        seq_len=64,
        dim_size=128,
        n_blocks=2,
        num_heads=4,
        dropout=0.1,
        lr=5e-4,
    )
    cb = pl.callbacks.ModelCheckpoint(
        dirpath="ckpt", save_top_k=3, monitor="val_loss", mode="min"
    )
    trainer = pl.Trainer(
        accelerator="mps",
        devices=1,
        max_epochs=10,
        val_check_interval=1000,
        callbacks=[cb],
        log_every_n_steps=10,
    )
    trainer.fit(model, dm)
