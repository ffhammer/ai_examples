import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models import SmallVit, VitLitModel  # use updated model + wrapper

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(0.1),
        transforms.ColorJitter(0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

train_set = datasets.MNIST("data", train=True, download=True, transform=transform)
val_set = datasets.MNIST("data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4)


vit = SmallVit(
    image_size=32,
    patches=4,
    num_classes=10,
    dim_size=256,
    n_blocks=2,
    n_channels=1,
    num_heads=4,
    dropout=0.1,
)
model = VitLitModel(vit, lr=3e-4)


if __name__ == "__main__":
    trainer = Trainer(
        max_epochs=10,
        accelerator="mps",
        log_every_n_steps=10,
        callbacks=[ModelCheckpoint(monitor="val_acc", mode="max")],
    )
    trainer.fit(model, train_loader, val_loader)
