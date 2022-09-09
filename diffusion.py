import argparse
import os
import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import torch.nn.functional as F
import torch.optim as optim

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

    x_t, E = forward(x_0, t)
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
        args.data_path, split="train", transform=transform, download=True
    )
    testdata = datasets.CelebA(
        args.data_path, split="test", transform=transform, download=True
    )
    validdata = datasets.CelebA(
        args.data_path, split="valid", transform=transform, download=True
    )

    trainloader = torch.utils.data.DataLoader(
        traindata, batch_size=batch_size, shuffle=True, drop_last=True
    )
    validloader = torch.utils.data.DataLoader(
        validdata, batch_size=batch_size, shuffle=True, drop_last=True
    )
    testloader = torch.utils.data.DataLoader(
        testdata, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return (trainloader, validloader, testloader)


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


def save_model():
    torch.save(model.state_dict(), args.model if args.model != None else "model.pth")


def train_model(train, validate):
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        print("-" * 32)
        print(f"Epoch {epoch + 1}")
        print("-" * 32)

        for step, batch in enumerate(train):
            optimizer.zero_grad()

            t = torch.randint(0, nsteps, (batch_size,), device=device).long()
            x_0 = batch[0].to(device)

            loss = L2Loss(model, x_0, t)
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"\tstep {step}, loss {loss:>.8f}")

                if step % 250 == 0:
                    validation_loss = validate_model(validate)
                    print(f"validation {epoch+1}/{step}: {validation_loss:>.8f}")

                    if step > 0 and step % 500 == 0:
                        save_model()


@torch.no_grad()
def validate_model(loader):
    total_loss = 0

    for (x_0, _) in loader:
        t = torch.randint(0, nsteps, (batch_size,), device=device).long()
        x_0 = x_0.to(device)

        total_loss += L2Loss(model, x_0, t).item()

    loss = total_loss / len(loader)

    return loss


def parse_args():
    parser = argparse.ArgumentParser("diffusion")

    parser.add_argument(
        "model", nargs="?", help="Load saved weights for model", type=str, default=None
    )
    parser.add_argument(
        "--generate",
        help="Use diffusion to synthesize an image given a model",
        action="store_true",
    )
    parser.add_argument(
        "--disable_dataloader", help="Disable loading datasets", action="store_true"
    )
    parser.add_argument(
        "--epochs",
        nargs="?",
        help="Number of epochs for training",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--batch", nargs="?", help="Set batch size", type=int, default=32
    )
    parser.add_argument(
        "--image_size",
        nargs="?",
        help="Set side length of square image",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--steps", nargs="?", help="Number of steps", type=int, default=300
    )
    parser.add_argument(
        "--test_forward_process",
        help="Test forward diffusion process",
        action="store_true",
    )
    parser.add_argument(
        "--parallel",
        help="Run model in parallel if multiple GPUs, defaults to all available GPUs",
        action="store_true",
    )
    parser.add_argument(
        "--data_path", help="Where to load/save datasets", type=str, default="./data"
    )
    parser.add_argument(
        "--headless", help="Disable matplotlib functions", action="store_true"
    )
    parser.add_argument(
        "--learning_rate",
        help="Set learning rate on Adam optimizer",
        type=float,
        default=1e-3,
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    batch_size = args.batch
    image_size = args.image_size
    nsteps = args.steps

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
    model = Net(image_size)

    if args.parallel:
        model = torch.nn.DataParallel(model)

    model.to(device)

    if not args.disable_dataloader:
        trainloader, validloader, testloader = create_dataloader(
            (image_size, image_size), batch_size
        )

    if args.model != None and os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model))

    if not args.headless:
        # FIXME: Refactor this block into different file; this looks ugly
        import matplotlib.pyplot as plt

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
                step = nsteps // 5

                for idx in range(0, nsteps, step):
                    t = torch.Tensor([idx]).type(torch.int64)
                    x_t, _ = forward(image, t)
                    images.append(x_t.detach().cpu())

                x = images[-1][None, ...].to(device)
            else:
                image = torch.randn((3, image_size, image_size)).to(device)
                step = nsteps // 10
                x = image[None, ...]

            for idx in reversed(range(nsteps)):
                t = torch.Tensor([idx]).type(torch.int64).to(device)
                x = denoise(x, t)

                if idx % step == 0:
                    images.append(torch.squeeze(x.detach().cpu()))

            grid = make_grid(images, nrow=10, normalize=True, value_range=(-1, 1))
            grid = grid.permute(1, 2, 0).numpy()

            plt.imshow(grid)
            plt.show()

        if args.test_forward_process:
            test_forward_process(trainloader)

        if args.generate:
            if args.disable_dataloader:
                generate()
            else:
                generate(testloader)

    if not args.disable_dataloader and not args.generate:
        train_model(trainloader, validloader)
