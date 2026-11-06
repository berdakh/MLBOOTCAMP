"""A small, explicit training loop - the parts every framework hides.

Level 3 uses this loop (and asks you to modify it) instead of a black-box
trainer, so that mixed precision, gradient accumulation, clipping,
checkpointing and evaluation are visible lines of code. Hugging Face's
``Trainer`` and Lightning do the same things; after this you will know what
their flags mean.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .device import amp_dtype, autocast, sync


@dataclass
class TrainConfig:
    epochs: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    grad_accum_steps: int = 1
    max_grad_norm: float | None = 1.0
    mixed_precision: bool = True
    log_every: int = 50
    max_steps: int | None = None  # stop early (smoke tests, budgets)
    checkpoint_dir: str | None = None


@dataclass
class TrainResult:
    steps: int
    seconds: float
    examples_per_second: float
    history: list[dict] = field(default_factory=list)


def linear_warmup_cosine(optimizer, warmup: int, total: int):
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train(
    model: torch.nn.Module,
    batches: Callable[[], Iterable[dict]],
    loss_fn: Callable[[torch.nn.Module, dict], torch.Tensor],
    device: torch.device,
    config: TrainConfig,
    evaluate: Callable[[torch.nn.Module], dict] | None = None,
    batch_size_of: Callable[[dict], int] = lambda b: len(next(iter(b.values()))),
) -> TrainResult:
    """Train ``model``. ``batches()`` returns a fresh iterable of dict batches per epoch.

    ``loss_fn(model, batch)`` must move the batch to ``device`` (or receive it
    already there) and return a scalar loss. Mixed precision uses fp16 + a
    GradScaler on GPUs without bf16 (e.g. T4), bf16 where supported, and is a
    no-op on CPU.
    """
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    try:
        steps_per_epoch = math.ceil(len(batches()) / config.grad_accum_steps)  # type: ignore[arg-type]
    except TypeError:
        steps_per_epoch = config.max_steps or 1000
    total = config.max_steps or steps_per_epoch * config.epochs
    scheduler = linear_warmup_cosine(optimizer, config.warmup_steps, total)
    use_amp = config.mixed_precision and amp_dtype(device) is not None
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp and amp_dtype(device) == torch.float16)

    history: list[dict] = []
    step, seen, running = 0, 0, 0.0
    sync(device)
    start = time.perf_counter()
    model.train()
    done = False
    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(batches()):
            with autocast(device, enabled=use_amp):
                loss = loss_fn(model, batch) / config.grad_accum_steps
            scaler.scale(loss).backward()
            running += loss.item() * config.grad_accum_steps
            seen += batch_size_of(batch)
            if (i + 1) % config.grad_accum_steps:
                continue
            if config.max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            if step % config.log_every == 0:
                history.append({"step": step, "epoch": epoch, "loss": running / config.log_every,
                                "lr": scheduler.get_last_lr()[0]})
                running = 0.0
            if config.max_steps and step >= config.max_steps:
                done = True
                break
        if evaluate is not None:
            model.eval()
            with torch.no_grad():
                history.append({"step": step, "epoch": epoch, **evaluate(model)})
            model.train()
        if config.checkpoint_dir:
            save_checkpoint(model, optimizer, scheduler, step, Path(config.checkpoint_dir))
        if done:
            break
    sync(device)
    seconds = time.perf_counter() - start
    return TrainResult(step, seconds, seen / seconds if seconds else float("nan"), history)


def save_checkpoint(model, optimizer, scheduler, step: int, directory: Path) -> Path:
    """Everything needed to *resume*, not just to predict: model, optimizer, scheduler, step."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step-{step:06d}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "step": step}, tmp)
    tmp.replace(path)  # atomic: a crash mid-save never leaves a corrupt "latest"
    (directory / "latest.json").write_text(json.dumps({"path": path.name, "step": step}))
    return path


def load_checkpoint(model, optimizer=None, scheduler=None, directory: Path | None = None,
                    map_location="cpu") -> int:
    latest = json.loads((Path(directory) / "latest.json").read_text())
    state = torch.load(Path(directory) / latest["path"], map_location=map_location, weights_only=True)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    return int(state["step"])


def config_dict(config: TrainConfig) -> dict:
    return asdict(config)
