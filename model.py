import math
import torch
from torch import nn

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, embed_dim, upconv=False):
        super().__init__()

        if upconv:
            conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            xform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            xform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.block1 = nn.Sequential(
            conv1,
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
        )

        self.time_block = nn.Sequential(nn.Linear(embed_dim, out_ch), nn.ReLU())

        self.block2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
            xform,
        )

    def forward(self, x, t):
        t = self.time_block(t)
        t = t[..., None, None]

        x = self.block1(x)
        x += t
        x = self.block2(x)

        return x


class Embedding(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.ndim = ndim

        if ndim % 2 != 0:
            raise ValueError("Positional encoding must have an even dimension.")

    def forward(self, t):
        device = t.device
        hdim = self.ndim // 2

        embedding = math.log(10000) / (hdim - 1)
        embedding = torch.exp(torch.arange(0, hdim, device=device) * -embedding)
        embedding = t[:, None] * embedding[None, :]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)

        return embedding


class Net(nn.Module):
    def __init__(self, image_size):
        super().__init__()

        nch = int(math.log2(image_size) - 2)
        channels = [2 ** (i + 6) for i in range(0, nch)]
        embed_dim = 32

        self.time_block = nn.Sequential(
            Embedding(embed_dim), nn.Linear(embed_dim, embed_dim), nn.ReLU()
        )
        self.conv0 = nn.Conv2d(3, channels[0], 3, padding=1)
        self.dconv = nn.ModuleList(
            [Block(channels[i], channels[i + 1], embed_dim) for i in range(nch - 1)]
        )
        self.uconv = nn.ModuleList(
            [
                Block(channels[::-1][i], channels[::-1][i + 1], embed_dim, upconv=True)
                for i in range(nch - 1)
            ]
        )

        self.conv1 = nn.Conv2d(channels[0], 3, 1)

    def forward(self, x, t):
        t = self.time_block(t)
        x = self.conv0(x)

        activations = []

        for block in self.dconv:
            x = block(x, t)
            activations.append(x)

        for block in self.uconv:
            activation = activations.pop()
            x = torch.cat((x, activation), dim=1)
            x = block(x, t)

        x = self.conv1(x)

        return x
