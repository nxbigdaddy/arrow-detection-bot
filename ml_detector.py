"""Lazy YOLO inference adapter used by the main GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dataset_pipeline import CLASSES, ROOT


DEFAULT_MODEL = ROOT / "models" / "arrow_detector" / "weights" / "best.pt"


@dataclass(frozen=True)
class MLDetection:
    x: int
    y: int
    dx: int
    dy: int
    rainbow: bool
    confidence: float
    class_id: int


class ArrowMLDetector:
    def __init__(self, model_path: Path = DEFAULT_MODEL) -> None:
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        from ultralytics import YOLO
        self.model_path = model_path
        self.model = YOLO(str(model_path))

    def predict(self, image, confidence: float = 0.35) -> list[MLDetection]:
        result = self.model.predict(image, conf=confidence, verbose=False)[0]
        output: list[MLDetection] = []
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        if result.boxes is None:
            return output
        for box in result.boxes:
            class_id = int(box.cls.item())
            score = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dx, dy = directions[class_id % 4]
            output.append(MLDetection(round((x1 + x2) / 2), round((y1 + y2) / 2),
                                      dx, dy, class_id >= 4, score, class_id))
        return sorted(output, key=lambda d: (d.rainbow, d.confidence), reverse=True)


def model_available() -> bool:
    return DEFAULT_MODEL.exists()
