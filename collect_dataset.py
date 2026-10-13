"""Continuously collect unique game frames while the user plays manually."""

from __future__ import annotations

import argparse
import time

from dataset_pipeline import capture_current_sample


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="一箭又一箭")
    parser.add_argument("--interval", type=float, default=0.8)
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()
    created = 0
    print("采集已启动。请手动玩游戏；按 Ctrl+C 停止。")
    try:
        while created < args.count:
            try:
                path, is_new = capture_current_sample(args.title)
                if is_new:
                    created += 1
                    print(f"[{created}/{args.count}] {path.name}")
            except RuntimeError as exc:
                print(f"等待正常关卡：{exc}")
            time.sleep(max(0.25, args.interval))
    except KeyboardInterrupt:
        pass
    print(f"采集结束，共保存 {created} 张新截图。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
