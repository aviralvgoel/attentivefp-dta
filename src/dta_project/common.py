"""Configuration, reproducibility, safe output handling and checkpoint helpers."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def merge(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in update.items():
        out[key] = (
            merge(out[key], value)
            if isinstance(value, dict) and isinstance(out.get(key), dict)
            else value
        )
    return out


def load_config(path: str | Path, overrides=(), _seen=None) -> dict:
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError(f"Configuration inheritance cycle at {path}")
    seen.add(path)
    contents = path.read_text()
    raw = json.loads(contents) if path.suffix.lower() == ".json" else yaml.safe_load(contents)
    if not isinstance(raw, dict):
        raise TypeError("Configuration must be a YAML mapping")
    parent = raw.pop("extends", None)
    base = load_config(path.parent / parent, _seen=seen) if parent else {}
    result = merge(base, raw)
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep:
            raise ValueError(f"Expected KEY=VALUE, got {override!r}")
        section = result
        pieces = key.split(".")
        for part in pieces[:-1]:
            if part not in section or not isinstance(section[part], dict):
                raise KeyError(f"Unknown override {key}")
            section = section[part]
        if pieces[-1] not in section:
            raise KeyError(f"Unknown override {key}")
        section[pieces[-1]] = yaml.safe_load(value)
    return result


def parser_for_training(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume", type=Path, help="Resume last.pt at the next epoch")
    return parser


def resolve_paths(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    # All relative paths are relative to the shell's project directory, never the YAML directory.
    cfg["root"] = str(Path(cfg.get("root", ".")).expanduser().resolve())
    for section, keys in (
        ("data", ("processed_dir", "split_file", "protein_cache")),
        ("training", ("output_dir",)),
        ("transfer", ("checkpoint",)),
    ):
        for key in keys:
            value = cfg.get(section, {}).get(key)
            if value:
                p = Path(value).expanduser()
                cfg[section][key] = str((Path(cfg["root"]) / p).resolve())
    return cfg


def choose_device(name="auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run python doctor.py --cuda before training.")
    device = torch.device(name)
    if device.type == "cuda":
        # Exercise a real kernel; get_arch_list alone does not account for all PTX fallback cases.
        probe = torch.ones((8, 8), device=device)
        (probe @ probe).sum().item()
    return device


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def environment():
    versions = {}
    for pkg in (
        "torch",
        "torch-geometric",
        "numpy",
        "pandas",
        "rdkit",
        "transformers",
        "scikit-learn",
        "pyarrow",
    ):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = None
    return {
        "python": sys.version,
        "packages": versions,
        "git_commit": git_commit,
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


class RunDirectory:
    """Exclusive writer. Never silently overwrite another experiment."""

    def __init__(self, path, resume=False):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock = self.path / ".writer.lock"
        if not resume and any(self.path.iterdir()):
            raise FileExistsError(
                f"Run directory is not empty: {self.path}. Use a new run name or --resume."
            )
        try:
            fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(
                f"Run is locked: {self.lock}. Check the recorded PID before removing a stale lock."
            ) from exc
        with os.fdopen(fd, "w") as stream:
            stream.write(str(os.getpid()))

    def __enter__(self):
        return self.path

    def __exit__(self, *args):
        self.lock.unlink(missing_ok=True)


def logger_for(path):
    logger = logging.getLogger("dta_project")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(path) / "train.log"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def atomic_save(payload, path):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def rng_state():
    n = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)
