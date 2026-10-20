"""Run all paper dataset groups using the shared evaluation CLI."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["semantic", "noc"], required=True)
    p.add_argument("--model", choices=["tf", "lite"], required=True)
    p.add_argument("--split", choices=["remaining", "development", "full"], default="remaining")
    p.add_argument("--output-root", type=Path, default=Path("outputs"))
    args, extra = p.parse_known_args()
    if args.task == "semantic":
        groups = [(d, "all") for d in ["SPIN", "PascalPart", "PartImageNet"]]
    else:
        groups = [(d, "all") for d in ["GrabCut", "Berkeley", "DAVIS", "SBD", "PascalVOC"]]
        groups += [("PascalPart", c) for c in ["head", "wheel", "torso", "all"]]
        groups += [("PartImageNet", c) for c in ["head", "body", "foot", "all"]]
    for dataset, category in groups:
        output = args.output_root / args.model / args.task
        if args.task == "semantic":
            output /= args.split
        output /= f"{dataset.lower()}_{category}"
        subprocess.run([
            sys.executable, "-m", "sgr_click.evaluation.cli", args.task,
            "--model", args.model, "--dataset", dataset, "--category", category,
            "--split", args.split, "--output", str(output), *extra,
        ], check=True)


if __name__ == "__main__":
    main()
