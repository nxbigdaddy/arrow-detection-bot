"""Prepare the annotated dataset and train a small YOLO arrow detector."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from dataset_pipeline import ROOT, YOLO_ROOT, export_yolo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="", help="例如 0、cpu；留空自动选择")
    args = parser.parse_args()

    counts = export_yolo()
    print("dataset:", counts)
    if counts["train"] < 10 or counts["objects"] < 100:
        raise SystemExit("训练数据不足：建议至少标注 10 张截图、100 个箭头；正式训练建议 200–500 张。")

    from ultralytics import YOLO
    model = YOLO(args.model)
    result = model.train(
        data=str(YOLO_ROOT / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device or None, project=str(ROOT / "models"),
        name="arrow_detector", exist_ok=True, degrees=0, translate=0.03,
        scale=0.20, fliplr=0.0, flipud=0.0, mosaic=0.4,
    )
    best = Path(result.save_dir) / "weights" / "best.pt"
    target = ROOT / "models" / "arrow_detector" / "weights" / "best.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if best.resolve() != target.resolve():
        shutil.copy2(best, target)
    print("best model:", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
