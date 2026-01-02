from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import time

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class TrainConfig:
    batch_size: int
    epochs: int
    lr: float
    optimizer: str
    population: int
    population_batch: int
    group_size: int
    sigma: float
    es_lr: float
    data_dir: str
    device: str


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int = 28 * 28, hidden_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


def build_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_ds = datasets.MNIST(config.data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(config.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def materialize_dataset(dataset: torch.utils.data.Dataset, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    inputs: list[torch.Tensor] = []
    targets: list[int] = []
    for x, y in dataset:
        inputs.append(x)
        targets.append(int(y))
    input_tensor = torch.stack(inputs)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    if device != "cpu":
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)
    return input_tensor, target_tensor


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, device: str) -> tuple[float, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def _sample_lora_noise_batch(
    model: TinyMLP,
    batch_size: int,
    group_size: int,
    device: str,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    if batch_size % 2 != 0:
        raise ValueError("Antithetic sampling requires an even batch size.")
    if group_size % 2 != 0:
        raise ValueError("Antithetic sampling requires an even group size.")
    if batch_size % group_size != 0:
        raise ValueError("Batch size must be divisible by group size.")
    noise: dict[str, torch.Tensor] = {}
    group_count = batch_size // group_size
    half = group_size // 2
    for name, layer in (("fc1", model.fc1), ("fc2", model.fc2)):
        out_dim, in_dim = layer.weight.shape
        a_half = torch.randn(group_count, half, out_dim, 1, device=device, generator=generator)
        b_half = torch.randn(group_count, half, in_dim, 1, device=device, generator=generator)
        bias_half = torch.randn(group_count, half, out_dim, device=device, generator=generator)
        a_full = torch.cat([a_half, -a_half], dim=1).reshape(batch_size, out_dim, 1)
        b_full = torch.cat([b_half, b_half], dim=1).reshape(batch_size, in_dim, 1)
        bias_full = torch.cat([bias_half, -bias_half], dim=1).reshape(batch_size, out_dim)
        noise[f"{name}.weight.A"] = a_full
        noise[f"{name}.weight.B"] = b_full
        noise[f"{name}.bias"] = bias_full
    return noise


def _eggroll_forward_batch(
    model: TinyMLP,
    inputs: torch.Tensor,
    noise: dict[str, torch.Tensor],
    sigma: float,
) -> torch.Tensor:
    x = model.flatten(inputs)
    scale = sigma

    w1 = model.fc1.weight
    b1 = model.fc1.bias
    a1 = noise["fc1.weight.A"]
    b1_noise = noise["fc1.bias"]
    b1_lora = noise["fc1.weight.B"]
    base1 = x @ w1.t() + b1
    low1 = torch.einsum("pi,pir->pr", x, b1_lora)
    low1 = torch.einsum("pr,por->po", low1, a1)
    h1 = base1 + scale * low1 + sigma * b1_noise
    h1 = model.relu(h1)

    w2 = model.fc2.weight
    b2 = model.fc2.bias
    a2 = noise["fc2.weight.A"]
    b2_noise = noise["fc2.bias"]
    b2_lora = noise["fc2.weight.B"]
    base2 = h1 @ w2.t() + b2
    low2 = torch.einsum("pi,pir->pr", h1, b2_lora)
    low2 = torch.einsum("pr,por->po", low2, a2)
    return base2 + scale * low2 + sigma * b2_noise


def train_epoch_eggroll(
    model: TinyMLP,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    config: TrainConfig,
    epoch_idx: int,
    sigma: float,
    es_lr: float,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pop = config.population
    group_size = config.group_size
    pop_batch = config.population_batch or pop
    if pop % 2 != 0:
        raise ValueError("Antithetic sampling requires an even population size.")
    if group_size % 2 != 0:
        raise ValueError("Antithetic sampling requires an even group size.")
    if pop % group_size != 0:
        raise ValueError("Population size must be divisible by group size.")
    if pop_batch % group_size != 0:
        raise ValueError("Population batch must be divisible by group size.")

    prompts_per_step = pop // group_size
    num_train = train_inputs.size(0)
    start = (epoch_idx * prompts_per_step) % num_train
    prompt_idx = torch.arange(start, start + prompts_per_step, device=train_targets.device)
    prompt_idx = prompt_idx % num_train
    inputs_unique = train_inputs[prompt_idx]
    targets_unique = train_targets[prompt_idx]
    repeated_idx = prompt_idx.repeat_interleave(group_size)
    inputs_pop = train_inputs[repeated_idx]
    targets_pop = train_targets[repeated_idx]

    base_seed = torch.randint(0, 2**31 - 1, (1,)).item()
    fitness_tensor = torch.empty(pop, device=config.device)
    noise_batches: list[dict[str, torch.Tensor]] = []
    batch_sizes: list[int] = []

    with torch.no_grad():
        for start in range(0, pop, pop_batch):
            batch_size = min(pop_batch, pop - start)
            generator = torch.Generator(device=config.device)
            generator.manual_seed(base_seed + start)
            noise_batch = _sample_lora_noise_batch(
                model, batch_size, group_size, config.device, generator
            )
            outputs = _eggroll_forward_batch(
                model, inputs_pop[start : start + batch_size], noise_batch, sigma
            )
            losses = F.cross_entropy(outputs, targets_pop[start : start + batch_size], reduction="none")
            fitness_tensor[start : start + batch_size] = -losses
            noise_batches.append(noise_batch)
            batch_sizes.append(batch_size)

    if not torch.isfinite(fitness_tensor).all():
        return 0.0, 0.0

    fitness_grouped = fitness_tensor.view(prompts_per_step, group_size)
    fitness_mean = fitness_grouped.mean(dim=1, keepdim=True)
    global_std = fitness_tensor.std(unbiased=False)
    if global_std < 1e-8:
        global_std = torch.tensor(1.0, device=config.device)
    fitness_norm = (fitness_grouped - fitness_mean) / (global_std + 1e-8)
    fitness_norm = fitness_norm.view(pop)

    num_pairs = pop // 2
    scale = es_lr / num_pairs
    with torch.no_grad():
        w1_update = torch.zeros_like(model.fc1.weight)
        b1_update = torch.zeros_like(model.fc1.bias)
        w2_update = torch.zeros_like(model.fc2.weight)
        b2_update = torch.zeros_like(model.fc2.bias)

        offset = 0
        for noise_batch, batch_size in zip(noise_batches, batch_sizes, strict=False):
            group_count = batch_size // group_size
            half_group = group_size // 2
            fitness_batch = fitness_norm[offset : offset + batch_size].view(
                group_count, group_size
            )
            f_plus = fitness_batch[:, :half_group]
            f_minus = fitness_batch[:, half_group:]
            coeff = (f_plus - f_minus) / 2.0
            coeff = coeff.reshape(-1)
            offset += batch_size

            a1 = noise_batch["fc1.weight.A"].view(group_count, group_size, -1, 1)[:, :half_group]
            b1_lora = noise_batch["fc1.weight.B"].view(group_count, group_size, -1, 1)[:, :half_group]
            w1_update.add_(
                torch.einsum(
                    "p,por,pir->oi",
                    coeff,
                    a1.reshape(-1, a1.size(2), 1),
                    b1_lora.reshape(-1, b1_lora.size(2), 1),
                )
            )
            b1_noise = noise_batch["fc1.bias"].view(group_count, group_size, -1)[:, :half_group]
            b1_update.add_(
                torch.einsum("p,po->o", coeff, b1_noise.reshape(-1, b1_noise.size(2)))
            )

            a2 = noise_batch["fc2.weight.A"].view(group_count, group_size, -1, 1)[:, :half_group]
            b2_lora = noise_batch["fc2.weight.B"].view(group_count, group_size, -1, 1)[:, :half_group]
            w2_update.add_(
                torch.einsum(
                    "p,por,pir->oi",
                    coeff,
                    a2.reshape(-1, a2.size(2), 1),
                    b2_lora.reshape(-1, b2_lora.size(2), 1),
                )
            )
            b2_noise = noise_batch["fc2.bias"].view(group_count, group_size, -1)[:, :half_group]
            b2_update.add_(
                torch.einsum("p,po->o", coeff, b2_noise.reshape(-1, b2_noise.size(2)))
            )

        model.fc1.weight.add_(w1_update, alpha=scale)
        model.fc1.bias.add_(b1_update, alpha=scale)
        model.fc2.weight.add_(w2_update, alpha=scale)
        model.fc2.bias.add_(b2_update, alpha=scale)

        outputs = model(inputs_unique)
        loss = F.cross_entropy(outputs, targets_unique)
        preds = outputs.argmax(dim=1)

    running_loss += loss.item() * inputs_unique.size(0)
    correct += (preds == targets_unique).sum().item()
    total += targets_unique.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Tiny MNIST baseline training loop.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, choices=("adam", "eggroll"), default="adam")
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--population-batch", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--es-lr", type=float, default=0.02)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    return TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        optimizer=args.optimizer,
        population=args.population,
        population_batch=args.population_batch,
        group_size=args.group_size,
        sigma=args.sigma,
        es_lr=args.es_lr,
        data_dir=args.data_dir,
        device=args.device,
    )


def main() -> None:
    config = parse_args()
    torch.manual_seed(7)
    train_loader, test_loader = build_dataloaders(config)
    train_inputs = None
    train_targets = None
    if config.optimizer == "eggroll":
        train_inputs, train_targets = materialize_dataset(train_loader.dataset, config.device)
    model = TinyMLP().to(config.device)
    optimizer = None
    if config.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.lr)

    start_time = time.monotonic()
    time_to_target = None
    target_acc = 0.90

    for epoch in range(1, config.epochs + 1):
        if config.optimizer == "eggroll":
            progress = (epoch - 1) / max(config.epochs, 1)
            sigma = config.sigma * (1.0 - progress)
            es_lr = config.es_lr * (1.0 - progress)
            train_loss, train_acc = train_epoch_eggroll(
                model, train_inputs, train_targets, config, epoch - 1, sigma, es_lr
            )
        else:
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, config.device)
        test_loss, test_acc = eval_epoch(model, test_loader, config.device)
        if time_to_target is None and test_acc >= target_acc:
            time_to_target = time.monotonic() - start_time
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )
    if time_to_target is None:
        elapsed = time.monotonic() - start_time
        print(f"time_to_{int(target_acc * 100)}pct=not_reached elapsed={elapsed:.2f}s")
    else:
        print(f"time_to_{int(target_acc * 100)}pct={time_to_target:.2f}s")


if __name__ == "__main__":
    main()
