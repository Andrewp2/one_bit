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
    steps_per_epoch: int
    sigma: float
    bias_sigma: float
    es_lr: float
    use_sigma_scaling: bool
    noise_scale_mode: str
    fitness_shaping: str
    fitness_baseline: str
    sanity_check: bool
    es_optimizer: str
    es_momentum: float
    es_weight_decay: float
    es_log_every: int
    sigma_schedule: str
    es_lr_schedule: str
    schedule_warmup_frac: float
    schedule_decay: float
    sigma_floor: float
    es_lr_floor: float
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
    bias_sigma: float,
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
        if bias_sigma > 0:
            bias_half = torch.randn(group_count, half, out_dim, device=device, generator=generator)
        else:
            bias_half = torch.zeros(group_count, half, out_dim, device=device)
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
    bias_sigma: float,
    noise_scale_mode: str,
) -> torch.Tensor:
    x = model.flatten(inputs)
    scale = sigma

    w1 = model.fc1.weight
    b1 = model.fc1.bias
    w1_scale = 1.0
    if noise_scale_mode == "fan_in":
        w1_scale = 1.0 / math.sqrt(w1.size(1))
    a1 = noise["fc1.weight.A"]
    b1_noise = noise["fc1.bias"]
    b1_lora = noise["fc1.weight.B"]
    base1 = x @ w1.t() + b1
    low1 = torch.einsum("pi,pir->pr", x, b1_lora)
    low1 = torch.einsum("pr,por->po", low1, a1)
    h1 = base1 + (scale * w1_scale) * low1 + bias_sigma * b1_noise
    h1 = model.relu(h1)

    w2 = model.fc2.weight
    b2 = model.fc2.bias
    w2_scale = 1.0
    if noise_scale_mode == "fan_in":
        w2_scale = 1.0 / math.sqrt(w2.size(1))
    a2 = noise["fc2.weight.A"]
    b2_noise = noise["fc2.bias"]
    b2_lora = noise["fc2.weight.B"]
    base2 = h1 @ w2.t() + b2
    low2 = torch.einsum("pi,pir->pr", h1, b2_lora)
    low2 = torch.einsum("pr,por->po", low2, a2)
    return base2 + (scale * w2_scale) * low2 + bias_sigma * b2_noise


def _centered_ranks(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.zeros_like(values)
    ranks = torch.argsort(torch.argsort(values))
    ranks = ranks.to(dtype=torch.float32)
    return ranks / (values.numel() - 1) - 0.5


def _schedule_value(
    base: float,
    step: int,
    total_steps: int,
    schedule: str,
    warmup_frac: float,
    floor: float,
    decay: float,
) -> float:
    if total_steps <= 1:
        return max(base, floor)
    progress = step / (total_steps - 1)
    if progress < warmup_frac:
        return max(base, floor)
    adj = (progress - warmup_frac) / max(1e-8, 1.0 - warmup_frac)
    if schedule == "constant":
        value = base
    elif schedule == "linear":
        value = base * (1.0 - adj)
    elif schedule == "cosine":
        value = base * (0.5 * (1.0 + math.cos(math.pi * adj)))
    elif schedule == "exp":
        value = base * math.exp(-decay * adj)
    else:
        value = base
    return max(value, floor)


def train_epoch_eggroll(
    model: TinyMLP,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    config: TrainConfig,
    epoch_idx: int,
    sigma: float,
    es_lr: float,
    es_optimizer: optim.Optimizer | None,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pop = config.population
    group_size = config.group_size
    pop_batch = config.population_batch or pop
    steps_per_epoch = max(1, config.steps_per_epoch)
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
    generator = torch.Generator(device=config.device)
    generator.manual_seed(epoch_idx + 7)

    for step in range(steps_per_epoch):
        prompt_idx = torch.randint(
            0, num_train, (prompts_per_step,), device=train_targets.device, generator=generator
        )
        inputs_unique = train_inputs[prompt_idx]
        targets_unique = train_targets[prompt_idx]
        repeated_idx = prompt_idx.repeat_interleave(group_size)
        inputs_pop = train_inputs[repeated_idx]
        targets_pop = train_targets[repeated_idx]

        base_seed = torch.randint(0, 2**31 - 1, (1,)).item()
        fitness_tensor = torch.empty(pop, device=config.device)

        with torch.no_grad():
            for start in range(0, pop, pop_batch):
                batch_size = min(pop_batch, pop - start)
                noise_gen = torch.Generator(device=config.device)
                noise_gen.manual_seed(base_seed + start)
                noise_batch = _sample_lora_noise_batch(
                    model, batch_size, group_size, config.bias_sigma, config.device, noise_gen
                )
                outputs = _eggroll_forward_batch(
                    model,
                    inputs_pop[start : start + batch_size],
                    noise_batch,
                    sigma,
                    config.bias_sigma,
                    config.noise_scale_mode,
                )
                losses = F.cross_entropy(outputs, targets_pop[start : start + batch_size], reduction="none")
                fitness_tensor[start : start + batch_size] = -losses

        if not torch.isfinite(fitness_tensor).all():
            continue

        if config.fitness_baseline == "per_prompt":
            fitness_grouped = fitness_tensor.view(prompts_per_step, group_size)
            fitness_adjusted = fitness_grouped - fitness_grouped.mean(dim=1, keepdim=True)
            fitness_adjusted = fitness_adjusted.view(pop)
        else:
            fitness_adjusted = fitness_tensor - fitness_tensor.mean()

        if config.fitness_shaping == "centered_ranks":
            fitness_shaped = _centered_ranks(fitness_adjusted)
        elif config.fitness_shaping == "zscore":
            std = fitness_adjusted.std(unbiased=False)
            if std < 1e-8:
                std = torch.tensor(1.0, device=config.device)
            fitness_shaped = fitness_adjusted / (std + 1e-8)
        else:
            fitness_shaped = fitness_adjusted

        num_pairs = pop // 2
        scale = es_lr / num_pairs
        with torch.no_grad():
            w1_update = torch.zeros_like(model.fc1.weight)
            b1_update = torch.zeros_like(model.fc1.bias)
            w2_update = torch.zeros_like(model.fc2.weight)
            b2_update = torch.zeros_like(model.fc2.bias)
            if config.sanity_check and step == 0:
                alt_w1_update = torch.zeros_like(model.fc1.weight)
                alt_w2_update = torch.zeros_like(model.fc2.weight)
            w1_scale = 1.0
            w2_scale = 1.0
            if config.noise_scale_mode == "fan_in":
                w1_scale = 1.0 / math.sqrt(model.fc1.weight.size(1))
                w2_scale = 1.0 / math.sqrt(model.fc2.weight.size(1))

            offset = 0
            for start in range(0, pop, pop_batch):
                batch_size = min(pop_batch, pop - start)
                noise_gen = torch.Generator(device=config.device)
                noise_gen.manual_seed(base_seed + start)
                noise_batch = _sample_lora_noise_batch(
                    model, batch_size, group_size, config.bias_sigma, config.device, noise_gen
                )
                group_count = batch_size // group_size
                half_group = group_size // 2
                fitness_batch = fitness_shaped[offset : offset + batch_size].view(
                    group_count, group_size
                )
                f_plus = fitness_batch[:, :half_group]
                f_minus = fitness_batch[:, half_group:]
                coeff_base = (f_plus - f_minus) / 2.0
                coeff_base = coeff_base.reshape(-1)
                offset += batch_size

                if config.use_sigma_scaling:
                    weight_coeff = coeff_base / max(sigma, 1e-8)
                    if config.bias_sigma > 0:
                        bias_coeff = coeff_base / max(config.bias_sigma, 1e-8)
                    else:
                        bias_coeff = torch.zeros_like(coeff_base)
                else:
                    weight_coeff = coeff_base
                    bias_coeff = coeff_base

                if config.sanity_check and step == 0 and config.use_sigma_scaling:
                    alt_weight_coeff = coeff_base / max(2.0 * sigma, 1e-8)

                a1 = noise_batch["fc1.weight.A"].view(group_count, group_size, -1, 1)[:, :half_group]
                b1_lora = noise_batch["fc1.weight.B"].view(group_count, group_size, -1, 1)[:, :half_group]
                w1_update.add_(
                    torch.einsum(
                        "p,por,pir->oi",
                        weight_coeff * w1_scale,
                        a1.reshape(-1, a1.size(2), 1),
                        b1_lora.reshape(-1, b1_lora.size(2), 1),
                    )
                )
                if config.sanity_check and step == 0 and config.use_sigma_scaling:
                    alt_w1_update.add_(
                        torch.einsum(
                            "p,por,pir->oi",
                            alt_weight_coeff * w1_scale,
                            a1.reshape(-1, a1.size(2), 1),
                            b1_lora.reshape(-1, b1_lora.size(2), 1),
                        )
                    )
                b1_noise = noise_batch["fc1.bias"].view(group_count, group_size, -1)[:, :half_group]
                if config.bias_sigma > 0:
                    b1_update.add_(
                        torch.einsum(
                            "p,po->o", bias_coeff, b1_noise.reshape(-1, b1_noise.size(2))
                        )
                    )

                a2 = noise_batch["fc2.weight.A"].view(group_count, group_size, -1, 1)[:, :half_group]
                b2_lora = noise_batch["fc2.weight.B"].view(group_count, group_size, -1, 1)[:, :half_group]
                w2_update.add_(
                    torch.einsum(
                        "p,por,pir->oi",
                        weight_coeff * w2_scale,
                        a2.reshape(-1, a2.size(2), 1),
                        b2_lora.reshape(-1, b2_lora.size(2), 1),
                    )
                )
                if config.sanity_check and step == 0 and config.use_sigma_scaling:
                    alt_w2_update.add_(
                        torch.einsum(
                            "p,por,pir->oi",
                            alt_weight_coeff * w2_scale,
                            a2.reshape(-1, a2.size(2), 1),
                            b2_lora.reshape(-1, b2_lora.size(2), 1),
                        )
                    )
                b2_noise = noise_batch["fc2.bias"].view(group_count, group_size, -1)[:, :half_group]
                if config.bias_sigma > 0:
                    b2_update.add_(
                        torch.einsum(
                            "p,po->o", bias_coeff, b2_noise.reshape(-1, b2_noise.size(2))
                        )
                    )

            if config.sanity_check and step == 0 and config.use_sigma_scaling:
                ratio_w1 = (w1_update.norm() / (alt_w1_update.norm() + 1e-8)).item()
                ratio_w2 = (w2_update.norm() / (alt_w2_update.norm() + 1e-8)).item()
                print(f"sanity_check sigma_scale_ratio_fc1={ratio_w1:.3f} fc2={ratio_w2:.3f}")

            if es_optimizer is None:
                model.fc1.weight.add_(w1_update, alpha=scale)
                if config.bias_sigma > 0:
                    model.fc1.bias.add_(b1_update, alpha=scale)
                model.fc2.weight.add_(w2_update, alpha=scale)
                if config.bias_sigma > 0:
                    model.fc2.bias.add_(b2_update, alpha=scale)
            else:
                for group in es_optimizer.param_groups:
                    group["lr"] = scale
                model.fc1.weight.grad = -w1_update
                model.fc2.weight.grad = -w2_update
                if config.bias_sigma > 0:
                    model.fc1.bias.grad = -b1_update
                    model.fc2.bias.grad = -b2_update
                else:
                    model.fc1.bias.grad = torch.zeros_like(model.fc1.bias)
                    model.fc2.bias.grad = torch.zeros_like(model.fc2.bias)
                es_optimizer.step()
                es_optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs_unique)
            loss = F.cross_entropy(outputs, targets_unique)
            preds = outputs.argmax(dim=1)

        running_loss += loss.item() * inputs_unique.size(0)
        correct += (preds == targets_unique).sum().item()
        total += targets_unique.size(0)

        if config.es_log_every and (step % config.es_log_every == 0):
            with torch.no_grad():
                fitness_min = fitness_tensor.min().item()
                fitness_max = fitness_tensor.max().item()
                fitness_mean = fitness_tensor.mean().item()
                fitness_std = fitness_tensor.std(unbiased=False).item()
                nonfinite = (~torch.isfinite(fitness_tensor)).float().mean().item()
                w1_norm = model.fc1.weight.norm().item()
                w2_norm = model.fc2.weight.norm().item()
                update_norm = (w1_update.norm() + w2_update.norm()).item()
            print(
                f"es_step={step} fitness_mean={fitness_mean:.4f} fitness_std={fitness_std:.4f} "
                f"fitness_min={fitness_min:.4f} fitness_max={fitness_max:.4f} "
                f"nonfinite={nonfinite:.4f} update_norm={update_norm:.4f} "
                f"w1_norm={w1_norm:.4f} w2_norm={w2_norm:.4f}"
            )

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
    parser.add_argument("--es-steps-per-epoch", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--bias-sigma", type=float, default=0.0)
    parser.add_argument("--es-lr", type=float, default=0.02)
    parser.add_argument("--no-sigma-scaling", action="store_true")
    parser.add_argument(
        "--noise-scale-mode",
        type=str,
        choices=("none", "fan_in"),
        default="none",
    )
    parser.add_argument(
        "--fitness-shaping",
        type=str,
        choices=("none", "zscore", "centered_ranks"),
        default="centered_ranks",
    )
    parser.add_argument(
        "--fitness-baseline",
        type=str,
        choices=("global", "per_prompt"),
        default="global",
    )
    parser.add_argument("--es-sanity-check", action="store_true")
    parser.add_argument(
        "--sigma-schedule",
        type=str,
        choices=("constant", "linear", "cosine", "exp"),
        default="constant",
    )
    parser.add_argument(
        "--es-lr-schedule",
        type=str,
        choices=("constant", "linear", "cosine", "exp"),
        default="constant",
    )
    parser.add_argument("--schedule-warmup-frac", type=float, default=0.0)
    parser.add_argument("--schedule-decay", type=float, default=2.0)
    parser.add_argument("--sigma-floor", type=float, default=0.0)
    parser.add_argument("--es-lr-floor", type=float, default=0.0)
    parser.add_argument(
        "--es-optimizer",
        type=str,
        choices=("none", "sgd", "adam"),
        default="none",
    )
    parser.add_argument("--es-momentum", type=float, default=0.9)
    parser.add_argument("--es-weight-decay", type=float, default=0.0)
    parser.add_argument("--es-log-every", type=int, default=0)
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
        steps_per_epoch=args.es_steps_per_epoch,
        sigma=args.sigma,
        bias_sigma=args.bias_sigma,
        es_lr=args.es_lr,
        use_sigma_scaling=not args.no_sigma_scaling,
        noise_scale_mode=args.noise_scale_mode,
        fitness_shaping=args.fitness_shaping,
        fitness_baseline=args.fitness_baseline,
        sanity_check=args.es_sanity_check,
        es_optimizer=args.es_optimizer,
        es_momentum=args.es_momentum,
        es_weight_decay=args.es_weight_decay,
        es_log_every=args.es_log_every,
        sigma_schedule=args.sigma_schedule,
        es_lr_schedule=args.es_lr_schedule,
        schedule_warmup_frac=args.schedule_warmup_frac,
        schedule_decay=args.schedule_decay,
        sigma_floor=args.sigma_floor,
        es_lr_floor=args.es_lr_floor,
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
    es_optimizer = None
    if config.optimizer == "eggroll" and config.es_optimizer != "none":
        if config.es_optimizer == "sgd":
            es_optimizer = optim.SGD(
                model.parameters(),
                lr=1.0,
                momentum=config.es_momentum,
                weight_decay=config.es_weight_decay,
            )
        elif config.es_optimizer == "adam":
            es_optimizer = optim.Adam(
                model.parameters(),
                lr=1.0,
                weight_decay=config.es_weight_decay,
            )

    start_time = time.monotonic()
    time_to_target = None
    target_acc = 0.90

    for epoch in range(1, config.epochs + 1):
        if config.optimizer == "eggroll":
            sigma = _schedule_value(
                config.sigma,
                epoch - 1,
                config.epochs,
                config.sigma_schedule,
                config.schedule_warmup_frac,
                config.sigma_floor,
                config.schedule_decay,
            )
            es_lr = _schedule_value(
                config.es_lr,
                epoch - 1,
                config.epochs,
                config.es_lr_schedule,
                config.schedule_warmup_frac,
                config.es_lr_floor,
                config.schedule_decay,
            )
            train_loss, train_acc = train_epoch_eggroll(
                model, train_inputs, train_targets, config, epoch - 1, sigma, es_lr, es_optimizer
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
