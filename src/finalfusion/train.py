"""Joint or single-task training for FinalFusion."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .data import build_task_loaders, load_json, set_loader_epoch, standardize_batch
from .losses import build_task_loss
from .model import ARCHITECTURE_ID as FINALFUSION_ARCH, TASKS, UnifiedFinalNet
from .rawprism import ARCHITECTURE_ID as RAWPRISM_ARCH, RawPrism
from .rawprism_v5 import ARCHITECTURE_ID as RAWPRISM_V5_ARCH, RawPrismV5
from .rawprism_v6 import ARCHITECTURE_ID as RAWPRISM_V6_ARCH, RawPrismV6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one task-routed IVIF/MEF/MFF fusion model.")
    parser.add_argument("--config", required=True, help="JSON configuration; see config.example.json.")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=None, help="Override config tasks.")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume.")
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(value: str) -> torch.device:
    device = torch.device(value or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def build_model(
    model_type: str,
    tasks: tuple[str, ...],
    width: int,
    use_film: bool = True,
    backbone: str = "dense",
    use_priors: bool = True,
    use_semantic: bool = False,
    use_rft: bool = True,
    backbone_kwargs: dict[str, Any] | None = None,
    visible_residual: bool = False,
    visible_residual_scale: float = 0.5,
) -> tuple[Any, str]:
    if model_type == "rawprism":
        return RawPrism(
            tasks=tasks,
            width=width,
            use_film=use_film,
            backbone=backbone,
            use_priors=use_priors,
            backbone_kwargs=backbone_kwargs,
        ), RAWPRISM_ARCH
    if model_type == "rawprism_v5":
        return RawPrismV5(
            tasks=tasks,
            use_semantic=use_semantic,
            backbone_kwargs=backbone_kwargs,
        ), RAWPRISM_V5_ARCH
    if model_type == "rawprism_v6":
        return RawPrismV6(
            tasks=tasks,
            use_semantic=use_semantic,
            use_rft=use_rft,
            use_priors=use_priors,
            backbone_kwargs=backbone_kwargs,
            visible_residual=visible_residual,
            visible_residual_scale=visible_residual_scale,
        ), RAWPRISM_V6_ARCH
    if model_type in ("finalfusion", "unified"):
        return UnifiedFinalNet(tasks=tasks, width=width), FINALFUSION_ARCH
    raise ValueError("model_type must be 'rawprism', 'rawprism_v5', 'rawprism_v6', or 'finalfusion'")


def mean_stats(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: sum(row.get(key, 0.0) for row in rows) / len(rows) for key in keys} if rows else {}


def scalar_stats(values: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().float().mean().cpu()) for key, value in values.items()}


def next_cycled(loader: Any, iterator: Any) -> tuple[Any, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_epoch(
    model: UnifiedFinalNet,
    loaders: dict[str, Any],
    criteria: dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    tasks: tuple[str, ...],
    steps_per_task: int,
    task_weights: dict[str, float],
    gradient_clip: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    parameter_clip: float | None,
) -> dict[str, dict[str, float]]:
    model.train()
    iterators = {task: iter(loaders[task]) for task in tasks}
    rows: dict[str, list[dict[str, float]]] = {task: [] for task in tasks}
    progress = tqdm(total=steps_per_task * len(tasks), desc="train", dynamic_ncols=True)
    for _ in range(steps_per_task):
        for task in tasks:
            batch, iterators[task] = next_cycled(loaders[task], iterators[task])
            source1, source2, metadata = standardize_batch(task, batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                output = model(task, source1, source2, **metadata)
            with torch.autocast(device_type=device.type, enabled=False):
                loss, stats = criteria[task](output, source1, source2, metadata)
                weighted_loss = loss * task_weights[task]
            if not torch.isfinite(weighted_loss):
                raise FloatingPointError("Non-finite {} training loss".format(task))
            scaler.scale(weighted_loss).backward()
            scaler.unscale_(optimizer)
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            if parameter_clip is not None:
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.clamp_(-parameter_clip, parameter_clip)
            row = scalar_stats(stats)
            row["weighted_loss"] = float(weighted_loss.detach().float().cpu())
            rows[task].append(row)
            progress.update(1)
            progress.set_postfix(task=task, loss="{:.5f}".format(row.get("loss", 0.0)))
    progress.close()
    return {task: mean_stats(rows[task]) for task in tasks}


@torch.no_grad()
def validate(
    model: UnifiedFinalNet,
    loaders: dict[str, Any],
    criteria: dict[str, torch.nn.Module],
    device: torch.device,
    tasks: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    model.eval()
    result = {}
    for task in tasks:
        rows = []
        for batch in tqdm(loaders[task], desc="val-{}".format(task), leave=False, dynamic_ncols=True):
            source1, source2, metadata = standardize_batch(task, batch, device)
            output = model(task, source1, source2, **metadata)
            loss, stats = criteria[task](output, source1, source2, metadata)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite {} validation loss".format(task))
            rows.append(scalar_stats(stats))
        if not rows:
            raise RuntimeError("{} validation loader produced no batches".format(task))
        result[task] = mean_stats(rows)
    return result


def validation_score(metrics: dict[str, dict[str, float]], task_weights: dict[str, float]) -> float:
    normalizer = sum(task_weights.values())
    return sum(task_weights[task] * metrics[task]["loss"] for task in metrics) / normalizer


def checkpoint_payload(
    epoch: int,
    model: UnifiedFinalNet,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    arch_id: str,
    config: dict[str, Any],
    best_score: float,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "architecture": arch_id,
        "architecture_config": model.architecture_config(),
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_validation_score": float(best_score),
        "config": config,
        "train": train_metrics,
        "validation": val_metrics,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    tasks = tuple(dict.fromkeys(args.tasks or config.get("tasks", TASKS)))
    if not tasks or set(tasks) - set(TASKS):
        raise ValueError("tasks must be a non-empty subset of {}".format(TASKS))
    missing_configs = set(tasks) - set(config.get("task_config", {}))
    if missing_configs:
        raise KeyError("Missing task_config entries: {}".format(sorted(missing_configs)))
    seed = int(config.get("seed", 20260816))
    set_seed(seed, bool(config.get("deterministic", True)))
    device = resolve_device(config.get("device", ""))
    output_dir = Path(config.get("output_dir", "finalfusion_runs/multitask")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "resolved_config.json", {**config, "tasks": list(tasks)})

    train_loaders, val_loaders, criteria = {}, {}, {}
    for task in tasks:
        task_config = config["task_config"][task]
        train_loaders[task], val_loaders[task] = build_task_loaders(task, task_config, seed)
        criteria[task] = build_task_loss(task, task_config, device)

    model_type = str(config.get("model_type", "rawprism"))
    model, arch_id = build_model(
        model_type,
        tasks,
        int(config.get("width", 44)),
        bool(config.get("use_film", True)),
        str(config.get("backbone", "dense")),
        bool(config.get("use_priors", True)),
        bool(config.get("use_semantic", False)),
        bool(config.get("use_rft", True)),
        config.get("backbone_kwargs"),
        bool(config.get("visible_residual", False)),
        float(config.get("visible_residual_scale", 0.5)),
    )
    model = model.to(device)
    optimizer_name = str(config.get("optimizer", "adamw")).lower()
    learning_rate = float(config.get("learning_rate", 1.0e-4))
    weight_decay = float(config.get("weight_decay", 0.0))
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=learning_rate, alpha=float(config.get("rmsprop_alpha", 0.6)),
            momentum=float(config.get("rmsprop_momentum", 0.15)), eps=float(config.get("rmsprop_epsilon", 1.0e-8)),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError("optimizer must be adamw, adam, or rmsprop")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config.get("lr_factor", 0.3)),
        patience=int(config.get("lr_patience", 4)),
        min_lr=float(config.get("min_learning_rate", 1.0e-6)),
    ) if config.get("use_scheduler", True) else None

    amp_enabled = bool(config.get("amp", False) and device.type == "cuda")
    amp_dtype = torch.bfloat16 if config.get("amp_dtype", "bfloat16") == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    task_weights = {task: float(config.get("task_weights", {}).get(task, 1.0)) for task in tasks}
    if any(weight <= 0 for weight in task_weights.values()):
        raise ValueError("all task weights must be positive")
    steps_per_task = int(config.get("steps_per_task", max(len(loader) for loader in train_loaders.values())))
    if steps_per_task <= 0:
        raise ValueError("steps_per_task must be positive")

    start_epoch, best_score = 1, float("inf")
    resume_path = Path(args.resume).resolve() if args.resume else None
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint.get("architecture") != arch_id:
            raise ValueError("Checkpoint is not a {} model".format(arch_id))
        checkpoint_tasks = tuple(checkpoint["architecture_config"]["tasks"])
        if checkpoint_tasks != tasks:
            # Task-subset fine-tune: load only the compatible tensors (shared
            # backbone/adapter/head for the requested tasks) and start a fresh
            # optimizer/schedule.  This is used for IVIF-only fine-tuning of a
            # jointly trained checkpoint.
            state = model.state_dict()
            compatible = {
                key: value for key, value in checkpoint["model"].items()
                if key in state and state[key].shape == value.shape
            }
            model.load_state_dict(compatible, strict=False)
            print("subset resume: loaded {}/{} tensors from {} (checkpoint tasks {}, requested {})".format(
                len(compatible), len(state), resume_path, checkpoint_tasks, tasks), flush=True)
        else:
            model.load_state_dict(checkpoint["model"])
            if config.get("reset_optimizer", False):
                # Fresh optimizer/schedule for a short joint fine-tune of a
                # converged checkpoint; keep the epoch counter so the run ends
                # after the requested number of additional epochs.
                start_epoch = int(checkpoint["epoch"]) + 1
                best_score = float("inf")
            else:
                optimizer.load_state_dict(checkpoint["optimizer"])
                if scheduler is not None and checkpoint.get("scheduler") is not None:
                    scheduler.load_state_dict(checkpoint["scheduler"])
                start_epoch = int(checkpoint["epoch"]) + 1
                best_score = float(checkpoint.get("best_validation_score", best_score))

    epochs = int(config.get("epochs", 60))
    history_path = output_dir / "history.jsonl"
    for epoch in range(start_epoch, epochs + 1):
        for loader in train_loaders.values():
            set_loader_epoch(loader, epoch - 1)
        train_metrics = train_epoch(
            model, train_loaders, criteria, optimizer, scaler, device, tasks, steps_per_task, task_weights,
            float(config.get("gradient_clip", 5.0)), amp_enabled, amp_dtype,
            float(config["parameter_clip"]) if config.get("parameter_clip") is not None else None,
        )
        val_metrics = validate(model, val_loaders, criteria, device, tasks)
        score = validation_score(val_metrics, task_weights)
        if scheduler is not None:
            scheduler.step(score)
        improved = score < best_score
        best_score = min(best_score, score)
        payload = checkpoint_payload(
            epoch, model, optimizer, scheduler, arch_id,
            {**config, "tasks": list(tasks), "model_type": model_type}, best_score, train_metrics, val_metrics
        )
        torch.save(payload, output_dir / "last.pt")
        if improved:
            torch.save(payload, output_dir / "best.pt")
        record = {
            "epoch": epoch,
            "score": score,
            "best_score": best_score,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": val_metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        print("Epoch {}/{} | score {:.6f} | best {:.6f} | lr {:.2e}".format(
            epoch, epochs, score, best_score, optimizer.param_groups[0]["lr"]
        ))


if __name__ == "__main__":
    main()
