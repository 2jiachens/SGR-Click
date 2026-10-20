"""Verify checkpoint bytes against the published SHA-256 manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["tf", "lite"], required=True)
    p.add_argument("--directory", type=Path, default=Path("checkpoints"))
    args = p.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / "checkpoints/manifest.json").read_text())
    names = ["cocolvis_vit_base.pth", "sgr_click_lite.pt" if args.model == "lite"
             else "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"]
    for name in names:
        path = args.directory / name
        expected = manifest[name]
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise SystemExit(f"Missing checkpoint or incorrect size: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected["sha256"]:
            raise SystemExit(f"Checksum mismatch: {path}")
        print(f"OK {name}")


if __name__ == "__main__":
    main()
