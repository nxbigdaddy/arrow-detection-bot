"""Dataset capture and YOLO export helpers for the arrow detector."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
import random
import shutil

import cv2

import arrow_bot


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
RAW_IMAGES = DATASET / "images" / "raw"
ANNOTATIONS = DATASET / "annotations"
YOLO_ROOT = DATASET / "yolo"

CLASSES = [
    "normal_up", "normal_right", "normal_down", "normal_left",
    "rainbow_up", "rainbow_right", "rainbow_down", "rainbow_left",
]


def ensure_directories() -> None:
    RAW_IMAGES.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS.mkdir(parents=True, exist_ok=True)


def image_digest(image) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("无法编码采集截图")
    return hashlib.sha256(encoded.tobytes()).hexdigest()[:16]


def capture_current_sample(title: str = "一箭又一箭") -> tuple[Path, bool]:
    """Capture the normalized play field and deduplicate by image content."""
    ensure_directories()
    hwnd = arrow_bot.find_window(title)
    frame = arrow_bot.capture_window(hwnd)
    arrow_bot.ensure_active_game(frame)
    region, offset = arrow_bot.game_region(frame)
    digest = image_digest(region)
    existing = list(RAW_IMAGES.glob(f"*_{digest}.png"))
    if existing:
        return existing[0], False
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = RAW_IMAGES / f"{stamp}_{digest}.png"
    if not cv2.imwrite(str(path), region):
        raise RuntimeError(f"保存截图失败：{path}")
    metadata = {
        "captured_at": datetime.now().isoformat(timespec="milliseconds"),
        "source_title": title,
        "offset": [int(offset[0]), int(offset[1])],
        "width": int(region.shape[1]),
        "height": int(region.shape[0]),
        "sha256_prefix": digest,
    }
    path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, True


def annotation_path(image_path: Path) -> Path:
    return ANNOTATIONS / f"{image_path.stem}.json"


def load_annotations(image_path: Path) -> list[dict]:
    path = annotation_path(image_path)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("objects", [])


def save_annotations(image_path: Path, objects: list[dict]) -> None:
    ensure_directories()
    payload = {"image": image_path.name, "classes": CLASSES, "objects": objects}
    annotation_path(image_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dataset_stats() -> dict[str, int]:
    ensure_directories()
    images = list(RAW_IMAGES.glob("*.png"))
    annotation_files = list(ANNOTATIONS.glob("*.json"))
    objects = 0
    valid_annotations = 0
    for path in annotation_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            objects += len(payload.get("objects", []))
            valid_annotations += 1
        except (OSError, json.JSONDecodeError):
            continue
    return {"screenshots": len(images), "annotated": valid_annotations,
            "objects": objects}


def export_yolo(validation_ratio: float = 0.2, seed: int = 20260724,
                box_size: int = 22) -> dict[str, int]:
    """Create a deterministic train/val dataset from JSON point annotations."""
    ensure_directories()
    if YOLO_ROOT.exists():
        shutil.rmtree(YOLO_ROOT)
    for split in ("train", "val"):
        (YOLO_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    images = sorted(RAW_IMAGES.glob("*.png"))
    annotated = [p for p in images if annotation_path(p).exists()]
    rng = random.Random(seed)
    rng.shuffle(annotated)
    val_count = max(1, round(len(annotated) * validation_ratio)) if len(annotated) > 1 else 0
    val_names = {p.name for p in annotated[:val_count]}
    counts = {"train": 0, "val": 0, "objects": 0}

    for image_path in annotated:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        split = "val" if image_path.name in val_names else "train"
        shutil.copy2(image_path, YOLO_ROOT / "images" / split / image_path.name)
        lines: list[str] = []
        for obj in load_annotations(image_path):
            class_id = int(obj["class_id"])
            x, y = float(obj["x"]), float(obj["y"])
            bw = min(box_size, width) / width
            bh = min(box_size, height) / height
            lines.append(f"{class_id} {x / width:.8f} {y / height:.8f} {bw:.8f} {bh:.8f}")
        (YOLO_ROOT / "labels" / split / f"{image_path.stem}.txt").write_text(
            "\n".join(lines), encoding="utf-8")
        counts[split] += 1
        counts["objects"] += len(lines)

    yaml = (
        f"path: {YOLO_ROOT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names: {json.dumps(dict(enumerate(CLASSES)), ensure_ascii=False)}\n"
    )
    (YOLO_ROOT / "data.yaml").write_text(yaml, encoding="utf-8")
    return counts
