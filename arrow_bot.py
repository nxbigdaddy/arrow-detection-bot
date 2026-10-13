"""Windows bot for the WeChat mini-game "一箭又一箭".

The default mode is deliberately read-only: it captures the game window, finds
arrow heads whose forward ray is clear, and writes a debug image.  Pass
``--auto`` only after checking that image.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageGrab


USER32 = ctypes.windll.user32
GDI32 = ctypes.windll.gdi32

USER32.PostMessageW.argtypes = (
    wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
)
USER32.PostMessageW.restype = wintypes.BOOL

PW_RENDERFULLCONTENT = 2
SW_RESTORE = 9
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_CANCELMODE = 0x001F
MK_LBUTTON = 0x0001


class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


@dataclass(frozen=True)
class Candidate:
    x: int
    y: int
    dx: int
    dy: int
    score: float
    clearance: int
    group: int = -1


def find_window(title: str) -> int:
    found: list[tuple[int, str]] = []
    current_pid = os.getpid()
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    @callback_type
    def callback(hwnd: int, _lparam: int) -> bool:
        length = USER32.GetWindowTextLengthW(hwnd)
        if length and USER32.IsWindowVisible(hwnd):
            pid = wintypes.DWORD()
            USER32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            # The GUI title also contains the game name. Never capture our own UI.
            if pid.value == current_pid:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            USER32.GetWindowTextW(hwnd, buf, length + 1)
            if title in buf.value:
                found.append((hwnd, buf.value))
        return True

    USER32.EnumWindows(callback, 0)
    if not found:
        raise RuntimeError(f"找不到标题包含 {title!r} 的可见窗口")
    exact = [hwnd for hwnd, window_title in found if window_title.strip() == title.strip()]
    return exact[0] if exact else found[0][0]


def window_rect(hwnd: int) -> tuple[int, int, int, int]:
    rect = RECT()
    if not USER32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise ctypes.WinError()
    return rect.left, rect.top, rect.right, rect.bottom


def capture_window(hwnd: int) -> np.ndarray:
    """Capture a window even when another window partly covers it."""
    if USER32.IsIconic(hwnd):
        USER32.ShowWindow(hwnd, SW_RESTORE)
        time.sleep(0.35)
    left, top, right, bottom = window_rect(hwnd)
    width, height = right - left, bottom - top
    window_dc = USER32.GetWindowDC(hwnd)
    memory_dc = GDI32.CreateCompatibleDC(window_dc)
    bitmap = GDI32.CreateCompatibleBitmap(window_dc, width, height)
    old = GDI32.SelectObject(memory_dc, bitmap)
    ok = USER32.PrintWindow(hwnd, memory_dc, PW_RENDERFULLCONTENT)

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD), ("biWidth", ctypes.c_long),
            ("biHeight", ctypes.c_long), ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", ctypes.c_long),
            ("biYPelsPerMeter", ctypes.c_long), ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    header = BITMAPINFOHEADER()
    header.biSize = ctypes.sizeof(header)
    header.biWidth = width
    header.biHeight = -height
    header.biPlanes = 1
    header.biBitCount = 32
    header.biCompression = 0
    raw = ctypes.create_string_buffer(width * height * 4)
    GDI32.GetDIBits(memory_dc, bitmap, 0, height, raw, ctypes.byref(header), 0)
    GDI32.SelectObject(memory_dc, old)
    GDI32.DeleteObject(bitmap)
    GDI32.DeleteDC(memory_dc)
    USER32.ReleaseDC(hwnd, window_dc)
    if not ok:
        # This fallback requires the window to be unobscured.
        return cv2.cvtColor(np.array(ImageGrab.grab((left, top, right, bottom))), cv2.COLOR_RGB2BGR)
    bgra = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
    return bgra[:, :, :3].copy()


def game_region(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Locate the large blue-grey play field instead of assuming fixed pixels."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # The play field is low-saturation blue with value around 60--100.
    mask = cv2.inRange(hsv, (95, 20, 35), (135, 100, 135))
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    choices = [stats[i] for i in range(1, count)
               if stats[i, cv2.CC_STAT_AREA] > image.shape[0] * image.shape[1] * 0.20]
    if not choices:
        raise RuntimeError("没有定位到游戏区域；请确保关卡画面已打开")
    x, y, w, h, _area = max(choices, key=lambda s: s[cv2.CC_STAT_AREA])
    # Remove HUD at top and bottom. The maze normally fills this inner field.
    top = y + max(70, int(h * 0.08))
    bottom = y + h - max(70, int(h * 0.08))
    return image[top:bottom, x:x + w], (x, top)


def ensure_active_game(image: np.ndarray) -> None:
    """Reject result/revive dialogs that cover the otherwise detectable field."""
    h, w = image.shape[:2]
    # Only inspect the central part of the right-side game panel. A revive/result
    # dialog is a large bright neutral rectangle; maze strokes occupy far less.
    panel = image[int(h * 0.22):int(h * 0.78), int(w * 0.48):int(w * 0.94)]
    if panel.size == 0:
        raise RuntimeError("游戏窗口尺寸异常")
    peak = np.max(panel, axis=2)
    spread = peak.astype(np.int16) - np.min(panel, axis=2).astype(np.int16)
    bright_neutral_ratio = float(np.mean((peak > 175) & (spread < 55)))
    if bright_neutral_ratio > 0.18:
        raise RuntimeError("检测到结算、复活或其他弹窗；请先进入正常关卡画面")


def arrow_mask(region: np.ndarray, include_color: bool = True) -> np.ndarray:
    # Arrow strokes are bright; include saturated pixels for rainbow arrows.
    b, g, r = cv2.split(region)
    bright = np.minimum(np.minimum(b, g), r)
    peak = np.maximum.reduce([b, g, r])
    neutral = np.maximum.reduce([b, g, r]) - np.minimum.reduce([b, g, r])
    normal = (bright > 145) & (neutral < 45)
    colored = include_color & (peak > 165) & (neutral > 45)
    mask = (normal | colored).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def make_template(direction: tuple[int, int], size: int) -> np.ndarray:
    """Filled arrow template matching the game's rendered arrow heads."""
    canvas = np.zeros((size, size), np.uint8)
    c = size // 2
    # Draw a right-facing, two-pixel shaft with a compact filled triangle.
    # The old thin-V template matched ordinary 90-degree corners too easily.
    cv2.line(canvas, (0, c), (size - 5, c), 255, 2)
    triangle = np.array([[size - 6, c - 4], [size - 1, c],
                         [size - 6, c + 4]], dtype=np.int32)
    cv2.fillConvexPoly(canvas, triangle, 255)
    dx, dy = direction
    rotations = {(1, 0): None, (0, 1): cv2.ROTATE_90_CLOCKWISE,
                 (-1, 0): cv2.ROTATE_180, (0, -1): cv2.ROTATE_90_COUNTERCLOCKWISE}
    rotation = rotations[(dx, dy)]
    return canvas if rotation is None else cv2.rotate(canvas, rotation)


def clear_distance(mask: np.ndarray, x: int, y: int, dx: int, dy: int,
                   corridor: int = 3) -> int:
    """Return distance to edge, or -1 if another stroke blocks the forward ray."""
    h, w = mask.shape
    distance = 0
    # Start beyond the arrow tip/wings.
    for step in range(7, max(h, w)):
        px, py = x + dx * step, y + dy * step
        if px < 1 or py < 1 or px >= w - 1 or py >= h - 1:
            return distance
        if dx:
            occupied = mask[max(0, py - corridor):py + corridor + 1, px]
        else:
            occupied = mask[py, max(0, px - corridor):px + corridor + 1]
        if np.any(occupied):
            return -1
        distance += 1
    return distance


def detect_candidates(mask: np.ndarray, threshold: float = 0.78,
                      require_clear: bool = True) -> list[Candidate]:
    candidates: list[Candidate] = []
    size = 13
    for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        template = make_template((dx, dy), size)
        response = cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED)
        # Local maxima provide non-maximum suppression before cross-direction NMS.
        maxima = response == cv2.dilate(response, np.ones((11, 11), np.uint8))
        ys, xs = np.where((response >= threshold) & maxima)
        for tx, ty in zip(xs, ys):
            # Tip offset in each rotated template.
            cx, cy = tx + size // 2, ty + size // 2
            tip_x, tip_y = cx + dx * (size // 2 - 1), cy + dy * (size // 2 - 1)
            clearance = clear_distance(mask, tip_x, tip_y, dx, dy)
            if clearance >= 0 or not require_clear:
                candidates.append(Candidate(tip_x, tip_y, dx, dy,
                                            float(response[ty, tx]), clearance))
    candidates.sort(key=lambda c: (c.score, c.clearance), reverse=True)
    kept: list[Candidate] = []
    for candidate in candidates:
        if all((candidate.x - old.x) ** 2 + (candidate.y - old.y) ** 2 > 12 ** 2
               for old in kept):
            kept.append(candidate)
    return kept


def candidate_is_rainbow(region: np.ndarray, candidate: Candidate) -> bool:
    """Estimate whether the shaft behind an arrow head is color-saturated."""
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    if candidate.dx == 0 and candidate.dy == 0:
        x, y = int(candidate.x), int(candidate.y)
        patch = hsv[max(0, y - 9):y + 10, max(0, x - 9):x + 10]
        if patch.size == 0:
            return False
        return int(np.count_nonzero((patch[:, :, 1] > 70)
                                    & (patch[:, :, 2] > 130))) >= 5
    points: list[tuple[int, int]] = []
    for step in range(4, 34):
        x = int(candidate.x - candidate.dx * step)
        y = int(candidate.y - candidate.dy * step)
        if 0 <= x < region.shape[1] and 0 <= y < region.shape[0]:
            points.append((x, y))
    if not points:
        return False
    saturated = sum(int(hsv[y, x, 1]) > 70 and int(hsv[y, x, 2]) > 130
                    for x, y in points)
    return saturated >= 4


def detect_line_candidates(region: np.ndarray, min_area: int = 24) -> list[Candidate]:
    """Return pressable points on substantial connected stroke components.

    Clicking/holding any part of a game line selects the entire arrow, so this
    is more robust than requiring its tiny arrow head to match a template.
    ``dx == dy == 0`` identifies these direction-independent candidates.
    """
    mask = arrow_mask(region)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    result: list[Candidate] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        ys, xs = np.where(labels == label)
        if len(xs) == 0:
            continue
        center_x, center_y = centroids[label]
        nearest = int(np.argmin((xs - center_x) ** 2 + (ys - center_y) ** 2))
        indices = [nearest, int(np.argmin(xs)), int(np.argmax(xs)),
                   int(np.argmin(ys)), int(np.argmax(ys))]
        chosen: list[tuple[int, int]] = []
        for rank, index in enumerate(indices):
            x, y = int(xs[index]), int(ys[index])
            if any((x - old_x) ** 2 + (y - old_y) ** 2 < 7 ** 2
                   for old_x, old_y in chosen):
                continue
            chosen.append((x, y))
            # Area-derived score is informational; rank keeps the midpoint first
            # when two samples have identical perimeter priority.
            score = min(0.99, 0.50 + area / 300.0) - rank * 0.002
            result.append(Candidate(x, y, 0, 0, score, 0, label))
    return result


def frame_fingerprint(region: np.ndarray) -> str:
    """Stable hash for caching probes while the maze has not changed."""
    mask = arrow_mask(region)
    tiny = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_AREA) > 90
    return hashlib.blake2b(np.packbits(tiny).tobytes(), digest_size=8).hexdigest()


def adaptive_hold_preview(hwnd: int, x: int, y: int,
                          durations: tuple[float, ...] = (0.14, 0.22, 0.34)
                          ) -> tuple[str, np.ndarray, dict[str, int]]:
    """Try the shortest preview hold first and escalate only on no response."""
    last_frame: np.ndarray | None = None
    last_stats: dict[str, int] = {"changed": 0, "red": 0, "grey": 0}
    for duration in durations:
        verdict, last_frame, last_stats = hold_preview(hwnd, x, y, duration)
        last_stats["hold_ms"] = int(duration * 1000)
        if verdict != "none":
            return verdict, last_frame, last_stats
    assert last_frame is not None
    return "none", last_frame, last_stats


def hold_preview(hwnd: int, x: int, y: int, hold_seconds: float = 0.45
                 ) -> tuple[str, np.ndarray, dict[str, int]]:
    """Long-press without committing and classify the game's preview.

    Returns ``safe`` for a new neutral-grey guide, ``blocked`` for a red guide,
    or ``none`` when the point did not behave like a selectable arrow.
    Coordinates are in the captured whole-window image.
    """
    before = capture_window(hwnd)
    rect = RECT()
    USER32.GetWindowRect(hwnd, ctypes.byref(rect))
    client_origin = wintypes.POINT(0, 0)
    USER32.ClientToScreen(hwnd, ctypes.byref(client_origin))
    client_x = int(x) - int(client_origin.x - rect.left)
    client_y = int(y) - int(client_origin.y - rect.top)
    packed = int(((client_y & 0xFFFF) << 16) | (client_x & 0xFFFF))
    USER32.PostMessageW(hwnd, WM_LBUTTONDOWN, MK_LBUTTON, wintypes.LPARAM(packed))
    time.sleep(max(0.18, hold_seconds))
    during = capture_window(hwnd)
    # Cancel the gesture before releasing, so validation itself never fires it.
    USER32.PostMessageW(hwnd, WM_CANCELMODE, 0, 0)
    USER32.PostMessageW(hwnd, WM_LBUTTONUP, 0, wintypes.LPARAM(packed))

    before_region, (ox, oy) = game_region(before)
    h, w = before_region.shape[:2]
    during_region = during[oy:oy + h, ox:ox + w]
    delta = np.max(cv2.absdiff(before_region, during_region), axis=2) > 22
    # Only classify changed pixels connected to this press point. A previous
    # red preview may still be fading elsewhere and must not poison the next
    # candidate's verdict.
    expanded = cv2.dilate(delta.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    _count, labels = cv2.connectedComponents(expanded)
    anchor_x, anchor_y = int(x - ox), int(y - oy)
    y1, y2 = max(0, anchor_y - 18), min(labels.shape[0], anchor_y + 19)
    x1, x2 = max(0, anchor_x - 18), min(labels.shape[1], anchor_x + 19)
    nearby_labels = np.unique(labels[y1:y2, x1:x2])
    nearby_labels = nearby_labels[nearby_labels != 0]
    relevant = delta & np.isin(labels, nearby_labels)
    b, g, r = cv2.split(during_region)
    red = relevant & (r > 125) & (r.astype(np.int16) > g.astype(np.int16) + 45) \
        & (r.astype(np.int16) > b.astype(np.int16) + 35)
    spread = np.maximum.reduce([b, g, r]).astype(np.int16) \
        - np.minimum.reduce([b, g, r]).astype(np.int16)
    neutral = relevant & (spread < 35) & (np.maximum.reduce([b, g, r]) > 75)
    stats = {"changed": int(np.count_nonzero(relevant)),
             "red": int(np.count_nonzero(red)),
             "grey": int(np.count_nonzero(neutral))}

    # Wait until this cancelled preview has substantially returned to its
    # original frame before the caller probes another candidate.
    if int(np.count_nonzero(delta)):
        settle_deadline = time.monotonic() + 0.45
        while time.monotonic() < settle_deadline:
            time.sleep(0.05)
            settled = capture_window(hwnd)
            settled_region = settled[oy:oy + h, ox:ox + w]
            residue = np.max(cv2.absdiff(before_region, settled_region), axis=2) > 22
            if int(np.count_nonzero(residue)) < 18:
                break
    if stats["red"] >= 12:
        return "blocked", during, stats
    if stats["grey"] >= 25 and stats["changed"] >= 35:
        return "safe", during, stats
    return "none", during, stats


def auto_adjust_zoom(hwnd: int, max_steps: int = 6) -> tuple[float, int]:
    """Keep the maze at a recognition-friendly 82--90% of play-field width."""
    actions = 0
    ratio = 0.0
    for _ in range(max_steps + 1):
        frame = capture_window(hwnd)
        ensure_active_game(frame)
        region, (ox, _oy) = game_region(frame)
        mask = arrow_mask(region)
        points = cv2.findNonZero(mask)
        if points is None:
            raise RuntimeError("缩放校准时没有检测到箭线")
        _x, _y, maze_w, _maze_h = cv2.boundingRect(points)
        ratio = maze_w / region.shape[1]
        if 0.82 <= ratio <= 0.90:
            break
        # Bottom controls are stable relative to the game panel: minus at 30%,
        # plus at 70%, both about 52 px above the window bottom.
        control_x = ox + region.shape[1] * (0.70 if ratio < 0.82 else 0.30)
        control_y = frame.shape[0] - 52
        physical_click_window(hwnd, int(control_x), int(control_y))
        actions += 1
        time.sleep(0.35)
    return ratio, actions


def annotate(region: np.ndarray, candidates: list[Candidate],
             verdicts: list[str] | None = None) -> np.ndarray:
    result = region.copy()
    for index, item in enumerate(candidates, 1):
        verdict = verdicts[index - 1] if verdicts and index <= len(verdicts) else "unknown"
        color = {"safe": (40, 220, 40), "blocked": (30, 30, 235),
                 "none": (145, 145, 145), "unknown": (0, 190, 255)}[verdict]
        label = {"safe": "OK", "blocked": "BLOCK", "none": "FALSE",
                 "unknown": "?"}[verdict]
        cv2.circle(result, (item.x, item.y), 9, color, 2)
        if item.dx or item.dy:
            cv2.arrowedLine(result, (item.x, item.y),
                            (item.x + item.dx * 28, item.y + item.dy * 28), color, 2)
        cv2.putText(result, f"{index}:{label}:{item.score:.2f}",
                    (item.x + 8, item.y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return result


def click_window(hwnd: int, x: int, y: int) -> None:
    """Send a click where x/y are coordinates in the captured whole-window image."""
    rect = RECT()
    USER32.GetWindowRect(hwnd, ctypes.byref(rect))
    client_origin = wintypes.POINT(0, 0)
    USER32.ClientToScreen(hwnd, ctypes.byref(client_origin))
    client_x = int(x) - int(client_origin.x - rect.left)
    client_y = int(y) - int(client_origin.y - rect.top)
    packed = int(((client_y & 0xFFFF) << 16) | (client_x & 0xFFFF))
    USER32.PostMessageW(hwnd, WM_LBUTTONDOWN, MK_LBUTTON, wintypes.LPARAM(packed))
    time.sleep(0.04)
    USER32.PostMessageW(hwnd, WM_LBUTTONUP, 0, wintypes.LPARAM(packed))


def physical_click_window(hwnd: int, x: int, y: int) -> None:
    """Foreground click for host-layer controls that ignore posted messages."""
    left, top, _right, _bottom = window_rect(hwnd)
    USER32.ShowWindow(hwnd, SW_RESTORE)
    USER32.SetForegroundWindow(hwnd)
    USER32.SetCursorPos(int(left + x), int(top + y))
    time.sleep(0.06)
    USER32.mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
    time.sleep(0.05)
    USER32.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP


def main() -> int:
    parser = argparse.ArgumentParser(description="一箭又一箭：截图识别与自动点击工具")
    parser.add_argument("--title", default="一箭又一箭", help="目标窗口标题的一部分")
    parser.add_argument("--auto", action="store_true", help="连续识别并点击（默认仅识别）")
    parser.add_argument("--once", action="store_true", help="配合 --auto 时只点击一次")
    parser.add_argument("--interval", type=float, default=1.15, help="每次点击后的等待秒数")
    parser.add_argument("--threshold", type=float, default=0.78, help="箭头模板匹配阈值")
    parser.add_argument("--max-clicks", type=int, default=300, help="单次运行最多点击数")
    parser.add_argument("--debug", type=Path, default=Path("debug-detection.png"))
    args = parser.parse_args()

    hwnd = find_window(args.title)
    clicks = 0
    while True:
        frame = capture_window(hwnd)
        region, (offset_x, offset_y) = game_region(frame)
        mask = arrow_mask(region)
        candidates = detect_candidates(mask, args.threshold)
        cv2.imwrite(str(args.debug), annotate(region, candidates))
        cv2.imwrite(str(args.debug.with_name(args.debug.stem + "-mask.png")), mask)
        print(f"检测到 {len(candidates)} 个路径畅通候选；调试图：{args.debug.resolve()}")
        if candidates:
            best = candidates[0]
            print(f"首选 tip=({best.x},{best.y}) dir=({best.dx},{best.dy}) "
                  f"score={best.score:.3f} clearance={best.clearance}")
        if not args.auto:
            return 0
        if not candidates:
            print("没有可靠候选，停止自动操作（可查看调试图或稍微降低 --threshold）。")
            return 2
        best = candidates[0]
        # Click a few pixels behind the head, safely on the arrow's shaft.
        click_window(hwnd, offset_x + best.x - best.dx * 5,
                     offset_y + best.y - best.dy * 5)
        clicks += 1
        print(f"已点击 {clicks}: window=({offset_x + best.x},{offset_y + best.y})")
        if args.once or clicks >= args.max_clicks:
            return 0
        time.sleep(max(0.25, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
