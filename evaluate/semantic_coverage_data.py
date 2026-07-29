from __future__ import annotations

import json
import zipfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_SAMPLES_PATH = REPO_ROOT / "evaluate" / "paper_samples.json"


@lru_cache(maxsize=1)
def load_paper_samples() -> dict[str, object]:
    return json.loads(PAPER_SAMPLES_PATH.read_text(encoding="utf-8"))


def read_image(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def decode_png(payload: bytes) -> np.ndarray:
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise RuntimeError("failed to decode PNG")
    return decoded[..., 0] if decoded.ndim == 3 else decoded


def decode_coco_segmentation(segmentation: object, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rle)
    elif isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return np.asarray(decoded, dtype=bool)


class SemanticHierarchyDataset:
    def __init__(self, args):
        self.args = args
        self.archive: zipfile.ZipFile | None = None
        self.spin_index: dict[str, dict[int, dict[str, object]]] | None = None

        if args.dataset in {"PartImageNet", "SPIN"}:
            self.archive = zipfile.ZipFile(args.partimagenet_zip)
            if args.dataset == "SPIN":
                self.spin_index = {}
                for level in ("subparts", "parts", "wholes"):
                    payload = json.loads(
                        (args.spin_annotations_dir / f"spin_test_{level}.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    self.spin_index[level] = {
                        int(item["id"]): item for item in payload["annotations"]
                    }

    def close(self) -> None:
        if self.archive is not None:
            self.archive.close()

    def samples(self) -> list[dict[str, object]]:
        if self.args.dataset == "PascalPart":
            return self._pascalpart_samples()
        if self.args.dataset == "PartImageNet":
            return self._partimagenet_samples()
        if self.args.dataset == "SPIN":
            return self._spin_samples()
        raise ValueError(f"unsupported dataset: {self.args.dataset}")

    def load(self, sample: dict[str, object]):
        if self.args.dataset == "PascalPart":
            return self._load_pascalpart(sample)
        if self.args.dataset == "PartImageNet":
            return self._load_partimagenet(sample)
        if self.args.dataset == "SPIN":
            return self._load_spin(sample)
        raise ValueError(f"unsupported dataset: {self.args.dataset}")

    def _pascalpart_samples(self) -> list[dict[str, object]]:
        samples = load_paper_samples()["pascalpart"]["all"]
        result = []
        for item in samples:
            sample_id = str(item["sample_id"])
            result.append(
                {
                    **item,
                    "sample_key": (
                        f"{sample_id}:{int(item['object_id'])}:{int(item['part_id'])}"
                    ),
                }
            )
        return result

    def _resolve_pascalpart_paths(self, sample_id: str) -> tuple[Path, Path]:
        annotation_path = self.args.pascalpart_annotations_dir / f"{sample_id}.mat"
        image_candidates = [
            self.args.pascalpart_images_dir / f"{sample_id}.jpg",
            self.args.pascalpart_images_dir / "train" / f"{sample_id}.jpg",
            self.args.pascalpart_images_dir / "val" / f"{sample_id}.jpg",
        ]
        image_path = next((path for path in image_candidates if path.exists()), image_candidates[0])
        return annotation_path, image_path

    def _load_pascalpart(self, sample: dict[str, object]):
        sample_id = str(sample["sample_id"])
        annotation_path, image_path = self._resolve_pascalpart_paths(sample_id)
        image = read_image(image_path)
        objects = loadmat(str(annotation_path))["anno"]["objects"][0, 0]
        object_id = int(sample["object_id"])
        part_id = int(sample["part_id"])
        part_mask = np.asarray(objects["parts"][0, object_id]["mask"][0, part_id]) > 0
        object_mask = np.asarray(objects["mask"][0, object_id]) > 0
        return image, [part_mask, object_mask], ["part", "whole_object"]

    def _partimagenet_samples(self) -> list[dict[str, object]]:
        return [
            {
                **row,
                "sample_key": (
                    f"{row['sample_id']}:{row['class_name']}:{row['source_values']}"
                ),
            }
            for row in load_paper_samples()["partimagenet"]["all"]
        ]

    def _load_partimagenet(self, sample: dict[str, object]):
        assert self.archive is not None
        split = str(sample["split"])
        sample_id = str(sample["sample_id"])
        image_payload = self.archive.read(f"PartImageNet/images/{split}/{sample_id}.JPEG")
        image_bgr = cv2.imdecode(
            np.frombuffer(image_payload, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image_bgr is None:
            raise RuntimeError(f"failed to decode image {sample_id}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        labels = decode_png(
            self.archive.read(f"PartImageNet/annotations/{split}/{sample_id}.png")
        )
        source_values = [
            int(value)
            for value in str(sample["source_values"]).replace(";", ",").split(",")
            if value.strip()
        ]
        part_mask = np.isin(labels, source_values)
        whole_mask = labels > 0
        return image, [part_mask, whole_mask], ["part", "whole_object"]

    def _spin_samples(self) -> list[dict[str, object]]:
        return list(load_paper_samples()["spin"]["samples"])

    def _load_spin(self, sample: dict[str, object]):
        assert self.archive is not None
        assert self.spin_index is not None
        split = str(sample["split"])
        sample_id = str(sample["file_name"])
        image_payload = self.archive.read(
            f"PartImageNet/images/{split}/{sample_id}.JPEG"
        )
        image_bgr = cv2.imdecode(
            np.frombuffer(image_payload, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image_bgr is None:
            raise RuntimeError(f"failed to decode SPIN image {sample_id}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        targets = [
            decode_coco_segmentation(
                self.spin_index[level][int(sample[f"{name}_ann_id"])]["segmentation"],
                image.shape[0],
                image.shape[1],
            )
            for level, name in (
                ("subparts", "subpart"),
                ("parts", "part"),
                ("wholes", "whole"),
            )
        ]
        return image, targets, ["subpart", "part", "whole_object"]
