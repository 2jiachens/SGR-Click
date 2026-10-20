"""Combine disjoint evaluation shards and write dataset/macro summaries."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .cli import atomic_json, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    seen = set()
    identity = None
    for run in args.runs:
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        current = tuple(json.dumps(manifest[k], sort_keys=True) for k in [
            "task", "model", "split", "protocol", "weight_sha256",
            "samples_sha256", "source_sha256",
        ])
        if identity is not None and identity != current:
            raise ValueError("Cannot merge different models, protocols, splits, sources, or weights.")
        identity = current
        for line in (run / "records.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (row["dataset"], row["category"], row["sample_key"])
            if key in seen:
                raise ValueError(f"Duplicate sample: {key}")
            seen.add(key)
            groups[key[:2]].append(row)
    summaries = {f"{d}/{c}": summarize(rows, manifest["task"]) for (d, c), rows in groups.items()}
    result = {"task": manifest["task"], "model": manifest["model"], "groups": summaries}
    if manifest["task"] == "semantic":
        result["macro"] = {}
        for metric in ["m_biou", "sor", "stagnant_window_rate", "jump_step_rate"]:
            expected = ["PascalPart/all", "PartImageNet/all"]
            if metric != "sor":
                expected.append("SPIN/all")
            values = [summaries[k][metric] for k in expected
                      if k in summaries and summaries[k][metric] is not None]
            result["macro"][metric] = float(np.mean(values)) if len(values) == len(expected) else None
    else:
        core = ["PascalPart/head", "PascalPart/torso", "PascalPart/wheel",
                "PartImageNet/head", "PartImageNet/body", "PartImageNet/foot"]
        if all(k in summaries for k in core):
            result["core_six_macro"] = {}
            for protocol in ["ten", "best101"]:
                if all(protocol in summaries[k] for k in core):
                    result["core_six_macro"][protocol] = {
                        str(t): float(np.mean([summaries[k][protocol][str(t)]["noc"] for k in core]))
                        for t in [85, 90]
                    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
