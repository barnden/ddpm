import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import os

from model import Net


def generate_constants(steps, initial=1e-4, final=2e-2):
    """
    Generates a linear schedule from [initial, final].
    Returns (alpha, beta), where beta is the linear schedule and alpha is 1 - beta
    """

    beta = torch.linspace(initial, final, steps)
    alpha = 1 - beta

    sqrt_alpha = torch.sqrt(alpha)

    # See Eq. 4 (DDPM) for formulation of alpha bar
    alpha_bar = torch.cumprod(alpha, dim=0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    # See Eq. 6 (DDPM) for formulation of beta tilde
    alpha_bar_prev = torch.cat((torch.tensor([1]), alpha_bar[:-1]))
    beta_tilde = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)

    return (
        alpha,
        beta,
        sqrt_alpha,
        alpha_bar,
        sqrt_alpha_bar,
        sqrt_one_minus_alpha_bar,
        beta_tilde,
    )


def get_index_from_list(vals, t, x_shape):
    # Helper function taken from DeepFindr's video (see README)
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())

    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward(x_0, t):
    """
    Generates a noised image :math:`x_t` given an :math:`x_0`.

    .. math::
        x_t = \\sqrt{\\bar{\\alpha}_t}\\ x_0 + \sqrt{1 - \\bar{\\alpha}_t}\\ \\varepsilon

    where :math:`\\varepsilon \\sim \\mathcal{N}(0, I)`.

    :param x_0: torch.Tensor: A 4D batch tensor (B, C, W, H)
    :param t: torch.Tensor: A 1D tensor of time steps to generate

    :return: (xt, E), A tuple where xt is the result of the forward process after t, and E is the noise applied onto the image.
    :rtype: (torch.Tensor, torch.Tensor)
    """

    A_t = get_index_from_list(sqrt_alpha_bar, t, x_0.shape).to(device)
    stdev = get_index_from_list(sqrt_one_minus_alpha_bar, t, x_0.shape).to(device)

    E = torch.randn_like(x_0).to(device)
    x_t = (x_0.to(device) * A_t) + (stdev * E)

    return (x_t, E)


def L2Loss(model, x_0, t):
    # Find L2 distance between predicted epsilon and actual epsilon.

    x_t, E = forward(x_0, t, device)
    predicted_E = model(x_t, t)

    return F.mse_loss(predicted_E, E)


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
    Generate x_t for t=0 to t=nsteps by increments of 10.
    Used to debug the forward process.
    """

    image = next(iter(dataloader))[0][0]
    image = torch.squeeze(image)

    images = []

    for idx in range(0, nsteps, 10):
        t = torch.Tensor([idx]).type(torch.int64)
        xt, _ = forward(image, t)
        images.append(xt.detach().cpu())

    grid = make_grid(images, nrow=5, normalize=True, value_range=(-1, 1))
    grid = grid.permute(1, 2, 0).numpy()

    plt.imshow(grid)
    plt.show()


@torch.no_grad()
def generate(dataloader=None):
    """
    Generate an x_T, then use reverse diffusion model to find x_0.

    If a dataloader is given, then we take the first image as x_0.
    This x_0 is passed into the forward process to generate x_T,
    which is then denoised into a predicted x'_0.
    """

    images = []
    if dataloader != None:
        image = next(iter(dataloader))[0][0]
        image = torch.squeeze(image)
        step = 60

        for idx in range(0, 300, step):
            t = torch.Tensor([idx]).type(torch.int64)
            x_t, _ = forward(image, t)
            images.append(x_t.detach().cpu())

        x = images[-1][None, ...].to(device)
    else:
        image = torch.randn((3, image_size, image_size)).to(device)
        step = 30
        x = image[None, ...]

    for idx in reversed(range(300)):
        t = torch.Tensor([idx]).type(torch.int64).to(device)
        x = denoise(x, t)

        if idx % step == 0:
            images.append(torch.squeeze(x.detach().cpu()))

    grid = make_grid(images, nrow=10, normalize=True, value_range=(-1, 1))
    grid = grid.permute(1, 2, 0).numpy()

    plt.imshow(grid)
    plt.show()


@torch.no_grad()
def denoise(x, t):
    """
    Given an image :math:`x_t`, denoise the image using the model to find :math:`x_{t - 1}`.

    From Eq. 10 in DDPM:

    .. math::
        \\mu_\\theta = \\frac{1}{\\sqrt{\\bar{\\alpha_t}}}\\left(x_t - \\frac{\\beta_t}{\\sqrt{1 - \\bar{\\alpha}_t}}\\varepsilon\\right)

    Using :math:`\\mu_\\theta`, we can then reconstruct :math:`x_{t - 1}`.
    """
    beta_t = get_index_from_list(beta, t, x.shape)
    sqrt_one_minus_alpha_bar_t = get_index_from_list(
        sqrt_one_minus_alpha_bar, t, x.shape
    )
    alpha_sqrt_t = get_index_from_list(sqrt_alpha, t, x.shape)

    E_theta = model(x, t)
    mu_theta = (x - beta_t * E_theta / sqrt_one_minus_alpha_bar_t) / alpha_sqrt_t

    if t == 0:
        return mu_theta

    noise = torch.randn_like(x)
    beta_tilde_t = get_index_from_list(beta_tilde, t, x.shape)
    predicted = mu_theta + torch.sqrt(beta_tilde_t) * noise

    return predicted


if __name__ == "__main__":
    batch_size = 32
    image_size = 64
    nsteps = 300

    (
        alpha,
        beta,
        sqrt_alpha,
        alpha_bar,
        sqrt_alpha_bar,
        sqrt_one_minus_alpha_bar,
        beta_tilde,
    ) = generate_constants(steps=nsteps)

    # Create/load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net(image_size).to(device)

    dataloader = create_dataloader((image_size, image_size), batch_size)

    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth"))

    # Set to True to view the progression of the forward process
    if False:
        test_forward_process(dataloader)

    # Set to True to generate an image from random noise
    if True:
        # Optional: Pass dataloader to take a random image, add noise, then reverse noise to generate a nearest neighbour image
        generate()
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(0, 100):
            print("-" * 32)
            print(f"Epoch {epoch + 1}")
            print("-" * 32)

            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()

                t = torch.randint(0, nsteps, (batch_size,), device=device).long()
                x_0 = batch[0].to(device)

                loss = L2Loss(model, x_0, t)
                loss.backward()
                optimizer.step()

                if step % 100 == 0:
                    print(f"\tstep {step}, loss {loss:>.8f}")

            torch.save(model.state_dict(), "model.pth")
