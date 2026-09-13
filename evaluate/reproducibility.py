"""Small, dependency-light helpers for reproducible paper experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed every RNG used by the maintained evaluation code."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str | None:
    """Return a streaming SHA-256 digest, or ``None`` for a missing artifact."""
    path = Path(path).expanduser()
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    exists = resolved.is_file()
    return {
        "path": str(resolved),
        "exists": exists,
        "size_bytes": resolved.stat().st_size if exists else None,
        "sha256": sha256_file(resolved) if exists else None,
    }


def _git_value(repo_root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def build_run_manifest(
    args: Namespace | Mapping[str, Any],
    repo_root: Path,
    artifacts: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Capture the code, environment, CLI and exact model/protocol identities."""
    repo_root = Path(repo_root).resolve()
    values = vars(args) if isinstance(args, Namespace) else dict(args)
    status = _git_value(repo_root, "status", "--porcelain")
    device = values.get("device", "cpu")
    device = torch.device(device)
    return {
        "schema_version": 1,
        "command": [str(item) for item in sys.argv],
        "arguments": _json_value(values),
        "code": {
            "git_commit": _git_value(repo_root, "rev-parse", "HEAD"),
            "git_dirty": bool(status) if status is not None else None,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda" and torch.cuda.is_available()
                else None
            ),
        },
        "artifacts": [artifact_record(path) for path in artifacts],
    }


def write_run_manifest(
    output_dir: Path,
    args: Namespace | Mapping[str, Any],
    repo_root: Path,
    artifacts: Sequence[str | Path] = (),
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            build_run_manifest(args, repo_root=repo_root, artifacts=artifacts),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
