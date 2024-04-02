from datasets import load_dataset
import torch
import pytorch_lightning as pl
from src.layers import *
import numpy as np
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from torch import nn

class SmallVit(nn.Module):

    def __init__(
        self, image_size, patches, num_classes = 1, dim_size=256, n_blocks=4, n_channels=3, num_heads=4,
    ):
        super().__init__()

        self.image_size = image_size
        self.patches = patches
        self.grid_size = image_size // patches
        self.n_grids = patches**2
        self.dim_size = dim_size
        self.n_blocks = n_blocks
        self.n_channels = n_channels

        assert self.image_size % self.patches == 0

        self._gen_index_blocks()

        self.blocks = nn.ModuleList([SelfAttentionBlock(dim_size, num_heads, 0.05)])

        self.pixel_projection = nn.Linear(
            self.n_channels * self.grid_size**2, self.dim_size, bias=False
        )

        self.class_token = torch.normal(
            0,
            0.05,
            size=(
                1,
                dim_size,
            ),
            requires_grad=True,
        )

        self.pos_embeddings = torch.normal(
            0, 0.05, size=(self.n_grids + 1, dim_size), requires_grad=True
        )
        
        self.head = nn.Linear(dim_size, num_classes)

    def _gen_index_blocks(
        self,
    ):

        x_block, y_block = [], []
        for i in range(self.patches):
            for j in range(self.patches):

                a = torch.arange(self.image_size)[
                    i * self.grid_size : (i + 1) * self.grid_size
                ]
                b = torch.arange(self.image_size)[
                    j * self.grid_size : (j + 1) * self.grid_size
                ]
                a, b = map(lambda x: x.flatten(), torch.meshgrid(a, b))
                x_block.append(a)
                y_block.append(b)

        self.x_index = torch.stack(x_block)
        self.y_index = torch.stack(y_block)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): of shape [bs, n_channels, image_size, image_size]
        """

        x = x[:, :, self.x_index, self.y_index]
        x = x.transpose(2, 1).reshape(
            -1, self.n_grids, self.n_channels * self.grid_size**2
        )

        x = self.pixel_projection(x)

        x = torch.cat(
            (self.class_token.unsqueeze(0).repeat((x.shape[0], 1, 1)), x), axis=1
        )
        
        x += self.pos_embeddings[None, :]
        
        for block in self.blocks:
            
            x = block(x, None)
            
        
        x = self.head(x[:,0])
        
        return x
        