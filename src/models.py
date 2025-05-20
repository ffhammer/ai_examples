import torch
from torch import nn
import pytorch_lightning as pl
from src.layers import SelfAttentionBlock


# modified SmallVit with conv patch embedding
class SmallVit(nn.Module):
    def __init__(
        self,
        image_size: int,
        patches: int,
        num_classes: int = 10,
        dim_size: int = 256,
        n_blocks: int = 4,
        n_channels: int = 3,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patches = patches
        self.grid_size = image_size // patches
        self.n_grids = patches * patches
        self.dim_size = dim_size

        # conv-based patch embedding
        self.conv_proj = nn.Conv2d(
            n_channels, dim_size, kernel_size=self.grid_size, stride=self.grid_size
        )

        self.class_token = nn.Parameter(torch.zeros(1, 1, dim_size))
        self.pos_embeddings = nn.Parameter(torch.zeros(1, self.n_grids + 1, dim_size))

        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(dim_size, num_heads, dropout_prob=dropout)
                for _ in range(n_blocks)
            ]
        )
        self.norm = nn.LayerNorm(dim_size)
        self.head = nn.Linear(dim_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B = x.size(0)
        x = self.conv_proj(x)  # (B, D, p, p)
        x = x.flatten(2).transpose(1, 2)  # (B, n_grids, D)
        cls = self.class_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)  # (B, n_grids+1, D)
        x = x + self.pos_embeddings  # (B, n_grids+1, D)
        for blk in self.blocks:
            x = blk(x, mask=None)
        x = self.norm(x)
        return self.head(x[:, 0])  # (B, num_classes)


# Lightning wrapper with loss & accuracy
class VitLitModel(pl.LightningModule):
    def __init__(self, model: SmallVit, lr: float = 1e-3) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
