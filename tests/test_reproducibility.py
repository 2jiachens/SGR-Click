from argparse import Namespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from evaluate.reproducibility import artifact_record, build_run_manifest, seed_everything


def test_seed_everything_repeats_numpy_and_torch_values():
    seed_everything(17)
    first_numpy = np.random.random(4)
    first_torch = torch.rand(4)
    seed_everything(17)
    np.testing.assert_array_equal(first_numpy, np.random.random(4))
    torch.testing.assert_close(first_torch, torch.rand(4), rtol=0, atol=0)


def test_artifact_record_contains_content_digest(tmp_path):
    artifact = tmp_path / "protocol.json"
    artifact.write_bytes(b'{"states": 11}')
    record = artifact_record(artifact)
    assert record["exists"] is True
    assert record["size_bytes"] == 14
    assert len(record["sha256"]) == 64


def test_manifest_serializes_paths_and_devices(tmp_path):
    manifest = build_run_manifest(
        Namespace(seed=0, output_dir=tmp_path, device=torch.device("cpu")),
        repo_root=tmp_path,
    )
    assert manifest["arguments"]["output_dir"] == str(tmp_path.resolve())
    assert manifest["arguments"]["device"] == "cpu"
    assert manifest["environment"]["device"] == "cpu"
