import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import math


def generate_schedule(steps, initial=1e-4, final=2e-2):
    """
    Generates a linear schedule from [initial, final].
    Returns (alpha, beta), where beta is the linear schedule and alpha is 1 - beta
    """

    beta = torch.linspace(initial, final, steps)
    alpha = 1 - beta

    return (alpha, beta)


def get_index_from_list(vals, t, x_shape):
    """
    Taken from DeepFindr's video (see README)
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())

    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward(x_0, t, device="cuda"):
    """
    Generates a noised image :math:`x_t` given an :math:`x_0`.

    .. math::
        x_t = \\sqrt{\\bar{\\alpha}_t}\\ x_0 + \sqrt{1 - \\bar{\\alpha}_t}\\ \\varepsilon

    where :math:`\\varepsilon \\sim \\mathcal{N}(0, I)`.

    :param x_0: torch.Tensor: A 4D batch tensor (B, C, W, H)
    :param t: torch.Tensor: A 1D tensor of time steps to generate
    :param device: str: Either "cpu" or "cuda"

    :return: (xt, E), A tuple where xt is the result of the forward process after t, and E is the noise applied onto the image.
    :rtype: (torch.Tensor, torch.Tensor)
    """

    A_t = get_index_from_list(alpha_bar_sqrt, t, x_0.shape).to(device)
    stdev = get_index_from_list(one_minus_alpha_bar_sqrt, t, x_0.shape).to(device)

    E = torch.randn_like(x_0).to(device)
    x_t = (x_0.to(device) * A_t) + (stdev * E)

    return (x_t, E)


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
    def __init__(self):
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


def L2Loss(model, x_0, t):
    x_t, E = forward(x_0, t, device)
    predicted_E = model(x_t, t)

    return F.mse_loss(predicted_E, E)


@torch.no_grad()
def sample_timestep(x, t):
    b_t = get_index_from_list(beta, t, x.shape)
    somact = get_index_from_list(one_minus_alpha_bar_sqrt, t, x.shape)
    recipr = get_index_from_list(alpha_bar_recip_sqrt, t, x.shape)

    model_mean = recipr * (x - b_t * model(x, t) / somact)

    if t == 0:
        return model_mean

    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    noise = torch.randn_like(x)
    mean = model_mean + torch.sqrt(posterior_variance_t) * noise

    return mean


def create_dataloader(image_size=(256, 256), batch_size=5):
    transform = transforms.Compose(
        # Resize images in dataset to image_size and normalise color values to [-1, 1]
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),
        ]
    )

    traindata = datasets.CelebA(
        "./data", split="train", transform=transform, download=True
    )
    testdata = datasets.CelebA(
        "./data", split="test", transform=transform, download=True
    )

    dataset = torch.utils.data.ConcatDataset([traindata, testdata])

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return dataloader


@torch.no_grad()
def test_forward_process(dataloader):
    """
    Generate x_t for t=0 to t=300 by increments of 10.
    Used to debug the forward process.
    """

    image = next(iter(dataloader))[0][0]
    image = torch.squeeze(image)

    images = []

    for idx in range(0, 300, 10):
        t = torch.Tensor([idx]).type(torch.int64)
        xt, _ = forward(image, t)
        images.append(xt.detach().cpu())

    grid = make_grid(images, nrow=5, normalize=True, value_range=(-1, 1))
    grid = grid.permute(1, 2, 0).numpy()

    plt.imshow(grid)
    plt.show()


@torch.no_grad()
def test_model(dataloader):
    """
    Generate an x_T, then use reverse diffusion model to find x_0
    """

    image = next(iter(dataloader))[0][0]
    image = torch.squeeze(image)

    images = []

    for idx in range(0, 300, 10):
        t = torch.Tensor([idx]).type(torch.int64)
        xt, _ = forward(image, t)
        images.append(xt.detach().cpu())

    x = images[-1][None, ...].to(device)

    for idx in range(300):
        t = torch.Tensor([idx]).type(torch.int64).to(device)
        x = denoise(x, t)

        if idx % 10 == 0:
            images.append(torch.squeeze(x.detach().cpu()))

    grid = make_grid(images, nrow=10, normalize=True, value_range=(-1, 1))
    grid = grid.permute(1, 2, 0).numpy()

    plt.imshow(grid)
    plt.show()


@torch.no_grad()
def denoise(x, t):
    betas_t = get_index_from_list(beta, t, x.shape)
    somact = get_index_from_list(one_minus_alpha_bar_sqrt, t, x.shape)
    srat = get_index_from_list(alpha_bar_recip_sqrt, t, x.shape)

    E_theta = model(x, t)
    mdl = srat * (x - betas_t * E_theta / somact)

    if t[0] == 0:
        return mdl

    noise = torch.randn_like(x)
    b_t1 = get_index_from_list(posterior_variance, t, x.shape)
    den = mdl + torch.sqrt(b_t1) * noise

    return den


if __name__ == "__main__":
    batch_size = 128
    image_size = 32

    alpha, beta = generate_schedule(steps=300)
    alpha_bar = torch.cumprod(alpha, dim=0)
    alpha_bar_sqrt = torch.sqrt(alpha_bar)
    one_minus_alpha_bar_sqrt = torch.sqrt(1.0 - alpha_bar)
    alpha_bar_recip_sqrt = torch.sqrt(1.0 / alpha)
    alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
    posterior_variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)

    # Set to True to view the progression of the forward process
    if False:
        test_forward_process(dataloader)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataloader = create_dataloader((image_size, image_size), batch_size)

    # Set to True to load "model.pth" to test the model
    if False:
        model.load_state_dict(torch.load("model.pth"))

        # test_model() takes a random image from the dataset
        # applies the forward process in T=300 steps
        # then does the denoising process.
        for i in range(10):
            test_model(dataloader)

    else:
        for epoch in range(18, 100):
            print("-" * 32)
            print(f"Epoch {epoch + 1}")
            print("-" * 32)

            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()

                t = torch.randint(0, 300, (batch_size,), device=device).long()
                x_0 = batch[0].to(device)

                loss = L2Loss(model, x_0, t)
                loss.backward()
                optimizer.step()

                if step % 100 == 0:
                    print(f"\tstep {step}, loss {loss:>.8f}")

            torch.save(model.state_dict(), "model.pth")
