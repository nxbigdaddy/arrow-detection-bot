"""Tkinter GUI for arrow_bot.py."""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

import arrow_bot
import dataset_pipeline
import ml_detector


class ArrowBotGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("一箭又一箭 - ML工具 v1.0")
        self.geometry("1040x760")
        self.minsize(860, 620)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.last_result: tuple[int, tuple[int, int], list[arrow_bot.Candidate]] | None = None
        self.probe_blacklist: dict[str, set[tuple[str, int]]] = {}
        self.ml_model: ml_detector.ArrowMLDetector | None = None

        self.title_var = tk.StringVar(value="一箭又一箭")
        self.threshold_var = tk.DoubleVar(value=0.78)
        self.interval_var = tk.DoubleVar(value=1.15)
        self.status_var = tk.StringVar(value="就绪：请先打开游戏关卡，然后点击“识别画面”")
        self.count_var = tk.StringVar(value="候选：--")
        self.model_var = tk.StringVar()

        self._build_ui()
        self._refresh_model_state()
        self.after(80, self._drain_events)

    def _build_ui(self) -> None:
        controls = ttk.Frame(self, padding=10)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="窗口标题").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(controls, textvariable=self.title_var, width=18).grid(
            row=0, column=1, padx=(5, 16), sticky=tk.W)
        ttk.Label(controls, text="识别阈值").grid(row=0, column=2, sticky=tk.W)
        ttk.Spinbox(controls, from_=0.30, to=0.80, increment=0.01,
                    textvariable=self.threshold_var, width=7).grid(
            row=0, column=3, padx=(5, 16), sticky=tk.W)
        ttk.Label(controls, text="点击间隔（秒）").grid(row=0, column=4, sticky=tk.W)
        ttk.Spinbox(controls, from_=0.3, to=5.0, increment=0.1,
                    textvariable=self.interval_var, width=7).grid(
            row=0, column=5, padx=(5, 16), sticky=tk.W)
        ttk.Label(controls, textvariable=self.count_var).grid(row=0, column=6, sticky=tk.W)

        buttons = ttk.Frame(self, padding=(10, 0, 10, 8))
        buttons.pack(fill=tk.X)
        self.detect_btn = ttk.Button(buttons, text="识别画面", command=self.detect_once)
        self.detect_btn.pack(side=tk.LEFT, padx=(0, 7))
        self.zoom_btn = ttk.Button(buttons, text="自动校准缩放", command=self.adjust_zoom)
        self.zoom_btn.pack(side=tk.LEFT, padx=7)
        self.step_btn = ttk.Button(buttons, text="安全点击一步", command=self.click_once)
        self.step_btn.pack(side=tk.LEFT, padx=7)
        self.start_btn = ttk.Button(buttons, text="开始自动运行", command=self.start_auto)
        self.start_btn.pack(side=tk.LEFT, padx=7)
        self.stop_btn = ttk.Button(buttons, text="停止", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=7)
        ttk.Button(buttons, text="采集训练截图", command=self.collect_sample).pack(side=tk.LEFT, padx=7)
        ttk.Button(buttons, text="打开标注工具", command=self.open_annotation).pack(side=tk.LEFT, padx=7)
        ttk.Button(buttons, text="开始训练", command=self.open_training).pack(side=tk.LEFT, padx=7)
        ttk.Button(buttons, text="打开调试图", command=self.open_debug).pack(side=tk.RIGHT)

        ttk.Label(self, textvariable=self.status_var, anchor=tk.W, padding=(10, 4)).pack(fill=tk.X)
        ttk.Label(self, textvariable=self.model_var, anchor=tk.W,
                  padding=(10, 0, 10, 4)).pack(fill=tk.X)

        pane = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=(3, 10))
        preview_frame = ttk.Labelframe(pane, text="实时识别预览", padding=8)
        log_frame = ttk.Labelframe(pane, text="运行日志", padding=8)
        pane.add(preview_frame, weight=4)
        pane.add(log_frame, weight=2)

        self.preview = ttk.Label(preview_frame, text="尚无预览", anchor=tk.CENTER)
        self.preview.pack(fill=tk.BOTH, expand=True)
        self.log = tk.Text(log_frame, width=35, wrap=tk.WORD, state=tk.DISABLED,
                           font=("Consolas", 9))
        scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _settings(self) -> tuple[str, float, float]:
        title = self.title_var.get().strip()
        if not title:
            raise ValueError("窗口标题不能为空")
        threshold = float(self.threshold_var.get())
        interval = float(self.interval_var.get())
        if not 0.2 <= threshold <= 0.9:
            raise ValueError("识别阈值应在 0.2 到 0.9 之间")
        if interval < 0.25:
            raise ValueError("点击间隔不能小于 0.25 秒")
        return title, threshold, interval

    def _set_running(self, running: bool) -> None:
        normal = tk.DISABLED if running else tk.NORMAL
        self.detect_btn.configure(state=normal)
        self.zoom_btn.configure(state=normal)
        model_ready = ml_detector.model_available()
        self.step_btn.configure(state=(normal if model_ready else tk.DISABLED))
        self.start_btn.configure(state=(normal if model_ready else tk.DISABLED))
        self.stop_btn.configure(state=tk.NORMAL if running else tk.DISABLED)

    def _refresh_model_state(self) -> None:
        ready = ml_detector.model_available()
        self.model_var.set(
            f"模型：{'已就绪，可自动运行' if ready else '尚未训练；自动点击已禁用'} | "
            f"{ml_detector.DEFAULT_MODEL}")
        self._set_running(False)

    def collect_sample(self) -> None:
        try:
            path, created = dataset_pipeline.capture_current_sample(self.title_var.get().strip())
            self._append_log(f"{'已采集' if created else '重复画面，未重复保存'}：{path}")
            self.status_var.set(f"训练截图：{path.name}")
        except Exception as exc:
            messagebox.showerror("采集失败", str(exc))

    def open_annotation(self) -> None:
        subprocess.Popen([sys.executable, str(dataset_pipeline.ROOT / "annotation_tool.py")],
                         cwd=str(dataset_pipeline.ROOT))

    def open_training(self) -> None:
        stats = dataset_pipeline.dataset_stats()
        if stats["annotated"] < 10 or stats["objects"] < 100:
            messagebox.showwarning(
                "训练数据不足",
                f"当前截图：{stats['screenshots']} 张\n"
                f"已标注：{stats['annotated']} 张\n"
                f"箭头标记：{stats['objects']} 个\n\n"
                "最低要求：10 张已标注截图、100 个箭头。\n"
                "请先运行采集工具，再打开标注工具完成标注。")
            return
        subprocess.Popen(["cmd.exe", "/c", str(dataset_pipeline.ROOT / "train.bat")],
                         cwd=str(dataset_pipeline.ROOT), creationflags=0x00000010)
        self._append_log("训练窗口已启动；训练完成后重启主 GUI 加载 best.pt。")

    def _launch(self, mode: str) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            settings = self._settings()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self.stop_event.clear()
        self._set_running(True)
        self.status_var.set("正在连接游戏窗口……")
        self.worker = threading.Thread(target=self._work, args=(mode, settings), daemon=True)
        self.worker.start()

    def detect_once(self) -> None:
        if not ml_detector.model_available():
            self.status_var.set("尚无训练模型：仅显示采集用候选，不允许自动点击")
        self._launch("detect")

    def click_once(self) -> None:
        if not ml_detector.model_available():
            messagebox.showwarning("模型未训练", "请先采集、标注并训练模型。")
            return
        self._launch("step")

    def adjust_zoom(self) -> None:
        self._launch("zoom")

    def start_auto(self) -> None:
        if not ml_detector.model_available():
            messagebox.showwarning("模型未训练", "没有 best.pt，自动运行已禁用。")
            return
        if not messagebox.askokcancel(
                "开始自动运行",
                "工具将自动点击识别到的箭头。\n\n"
                "建议先用“识别画面”检查绿色 1 号候选是否正确。是否继续？"):
            return
        self._launch("auto")

    def stop(self) -> None:
        self.stop_event.set()
        self.status_var.set("正在停止，请等待当前识别完成……")
        self.stop_btn.configure(state=tk.DISABLED)

    def _recognize(self, title: str, threshold: float):
        hwnd = arrow_bot.find_window(title)
        frame = arrow_bot.capture_window(hwnd)
        arrow_bot.ensure_active_game(frame)
        region, offset = arrow_bot.game_region(frame)
        mask = arrow_bot.arrow_mask(region)
        if ml_detector.model_available():
            if self.ml_model is None:
                self.ml_model = ml_detector.ArrowMLDetector()
            detections = self.ml_model.predict(region)
            candidates = [
                arrow_bot.Candidate(d.x, d.y, d.dx, d.dy, d.confidence, 0,
                                    -3 if d.rainbow else -2)
                for d in detections
            ]
            candidates.sort(key=lambda c: (c.group == -3, c.score), reverse=True)
            marked = arrow_bot.annotate(region, candidates)
            cv2.imwrite("debug-detection-mask.png", mask)
            return hwnd, offset, candidates, marked, 0.35, region

        # Dataset-building fallback only. Automatic clicking is disabled until
        # a trained model exists, so heuristic candidates cannot fire arrows.
        # Arrow antialiasing changes slightly after every removal. Start with the
        # requested strict threshold, then gradually relax it instead of stopping
        # simply because the remaining arrow heads score a few percent lower.
        # Search two bands below the strict display threshold. Lower-scoring
        # real arrows can otherwise remain hidden when all strict candidates
        # are blocked. Long-press validation keeps this wider search safe.
        floor = max(0.50, threshold - 0.12)
        levels: list[float] = []
        level = threshold
        while level >= floor - 0.001:
            levels.append(round(level, 3))
            level -= 0.02
        # Use the lowest adaptive level, then let the game's own long-press
        # preview reject false/blocked candidates. Stopping at the first strict
        # level can leave us with only blocked arrows while hiding a valid,
        # slightly lower-scoring arrow.
        used_threshold = levels[-1]
        line_candidates = arrow_bot.detect_line_candidates(region)
        arrow_candidates = arrow_bot.detect_candidates(
            mask, used_threshold, require_clear=False)
        # Template candidates close to a line-component press point add no
        # coverage, so remove those duplicates. Remaining templates are fallback.
        arrow_candidates = [
            c for c in arrow_candidates
            if all((int(c.x) - int(line.x)) ** 2 + (int(c.y) - int(line.y)) ** 2 > 10 ** 2
                   for line in line_candidates)
        ]
        candidates = line_candidates + arrow_candidates
        points = cv2.findNonZero(mask)
        if points is not None:
            bx, by, bw, bh = cv2.boundingRect(points)
        else:
            bx, by, bw, bh = 0, 0, region.shape[1], region.shape[0]

        def perimeter_distance(c: arrow_bot.Candidate) -> int:
            return min(abs(int(c.x) - bx), abs(int(c.x) - (bx + bw - 1)),
                       abs(int(c.y) - by), abs(int(c.y) - (by + bh - 1)))

        # Rainbow first. For ordinary arrows, prefer candidates on the current
        # maze's outer boundary; these are much more likely to be removable.
        candidates.sort(
            key=lambda c: (-int(arrow_bot.candidate_is_rainbow(region, c)),
                           -int(c.dx == 0 and c.dy == 0),
                           -int(c.clearance >= 0), perimeter_distance(c), -c.score))
        marked = arrow_bot.annotate(region, candidates)
        cv2.imwrite("debug-detection-mask.png", mask)
        return hwnd, offset, candidates, marked, used_threshold, region

    def _work(self, mode: str, settings: tuple[str, float, float]) -> None:
        title, threshold, interval = settings
        clicks = 0
        empty_retries = 0
        last_clicked: arrow_bot.Candidate | None = None
        try:
            if mode == "zoom":
                hwnd = arrow_bot.find_window(title)
                ratio, actions = arrow_bot.auto_adjust_zoom(hwnd)
                self.events.put(("log", f"缩放校准完成：迷宫宽度占比 {ratio:.1%}，"
                                        f"调节 {actions} 次。"))
                hwnd, offset, candidates, marked, used_threshold, raw_region = self._recognize(
                    title, threshold)
                self.events.put(("preview", marked))
                self.events.put(("count", len(candidates)))
                return
            while not self.stop_event.is_set():
                hwnd, offset, candidates, marked, used_threshold, raw_region = self._recognize(
                    title, threshold)
                self.last_result = (hwnd, offset, candidates)
                frame_key = arrow_bot.frame_fingerprint(raw_region)
                blacklist = self.probe_blacklist.setdefault(frame_key, set())
                # Keep memory bounded across completed levels.
                if len(self.probe_blacklist) > 20:
                    oldest = next(iter(self.probe_blacklist))
                    if oldest != frame_key:
                        self.probe_blacklist.pop(oldest, None)
                self.events.put(("count", len(candidates)))
                if candidates:
                    empty_retries = 0
                    best = candidates[0]
                    rainbow_count = sum(arrow_bot.candidate_is_rainbow(raw_region, c)
                                        for c in candidates)
                    line_count = sum(c.dx == 0 and c.dy == 0 for c in candidates)
                    self.events.put(("log", f"候选 {len(candidates)} 个"
                                            f"（线段 {line_count}，"
                                            f"七彩 {rainbow_count}）；首选 "
                                            f"({best.x},{best.y}) 方向({best.dx},{best.dy}) "
                                            f"置信度 {best.score:.3f} "
                                            f"(本次阈值 {used_threshold:.2f})"))
                else:
                    if mode == "auto" and empty_retries < 6:
                        empty_retries += 1
                        self.events.put(("log", f"暂未找到候选，等待动画后重试 "
                                                f"({empty_retries}/6)……"))
                        if self.stop_event.wait(0.6):
                            break
                        continue
                    self.events.put(("log", "连续未找到可靠且路径畅通的候选，已停止。"))
                    break

                if mode == "detect":
                    verdicts: list[str] = []
                    probed: list[arrow_bot.Candidate] = []
                    self.events.put(("status", f"已找到 {len(candidates)} 个视觉候选，"
                                               "正在执行长按验证……"))
                    for index, probe in enumerate(candidates, 1):
                        if self.stop_event.is_set():
                            break
                        cache_key = (("group", probe.group) if probe.group >= 0
                                     else ("point", (int(probe.x) << 16) | int(probe.y)))
                        if cache_key in blacklist:
                            continue
                        px = offset[0] + probe.x - probe.dx * 5
                        py = offset[1] + probe.y - probe.dy * 5
                        verdict, _hold_frame, stats = arrow_bot.adaptive_hold_preview(
                            hwnd, px, py)
                        probed.append(probe)
                        verdicts.append(verdict)
                        self.events.put(("log", f"只读长按验证 {index}: {verdict} "
                                                f"grey={stats['grey']} red={stats['red']} "
                                                f"hold={stats.get('hold_ms', 0)}ms"))
                        if verdict == "blocked" and probe.group >= 0:
                            blacklist.add(("group", probe.group))
                        elif verdict in ("blocked", "none"):
                            blacklist.add(cache_key)
                        self.events.put(("status", f"正在验证候选 {index}/{len(candidates)}"))
                        if verdict == "safe":
                            # We already have a valid next move. Leave the rest
                            # yellow instead of spending valuable level time.
                            self.events.put(("status", f"已找到可行箭头：候选 {index}，"
                                                       "绿色 OK"))
                            break
                    # Only show candidates actually tested. Hundreds of yellow
                    # raw matches obscure the single useful green result.
                    verified = arrow_bot.annotate(raw_region, probed, verdicts)
                    cv2.imwrite("debug-detection.png", verified)
                    self.events.put(("preview", verified))
                    safe_count = verdicts.count("safe")
                    blocked_count = verdicts.count("blocked")
                    self.events.put(("log", f"验证汇总：可行 {safe_count}，受阻 "
                                            f"{blocked_count}，误识别 {verdicts.count('none')}。"))
                    break

                self.events.put(("preview", marked))

                # Ask the game itself: a long press draws a grey guide if the
                # arrow can leave, red if blocked, and nothing for false matches.
                selected = None
                for index, probe in enumerate(candidates, 1):
                    if self.stop_event.is_set():
                        break
                    cache_key = (("group", probe.group) if probe.group >= 0
                                 else ("point", (int(probe.x) << 16) | int(probe.y)))
                    if cache_key in blacklist:
                        continue
                    px = offset[0] + probe.x - probe.dx * 5
                    py = offset[1] + probe.y - probe.dy * 5
                    verdict, preview_frame, stats = arrow_bot.adaptive_hold_preview(
                        hwnd, px, py)
                    self.events.put(("preview", preview_frame[
                        offset[1]:offset[1] + marked.shape[0],
                        offset[0]:offset[0] + marked.shape[1]]))
                    rainbow = arrow_bot.candidate_is_rainbow(raw_region, probe)
                    self.events.put(("log", f"长按验证 {index}: {verdict} "
                                            f"grey={stats['grey']} red={stats['red']}"
                                            f" hold={stats.get('hold_ms', 0)}ms"
                                            f"{' [七彩]' if rainbow else ''}"))
                    if verdict == "blocked" and probe.group >= 0:
                        blacklist.add(("group", probe.group))
                    elif verdict in ("blocked", "none"):
                        blacklist.add(cache_key)
                    self.events.put(("status", f"自动运行：长按验证 {index}/{len(candidates)}"))
                    if verdict == "safe":
                        selected = probe
                        verified = arrow_bot.annotate(raw_region, [probe], ["safe"])
                        cv2.imwrite("debug-detection.png", verified)
                        self.events.put(("preview", verified))
                        self.events.put(("status", f"灰线验证通过：准备点击候选 {index}"))
                        break
                    time.sleep(0.08)
                if selected is None:
                    self.events.put(("log", "候选均未通过灰线验证，已停止且未执行点击。"))
                    break
                best = selected
                if (last_clicked is not None
                        and best.dx == last_clicked.dx and best.dy == last_clicked.dy
                        and (best.x - last_clicked.x) ** 2
                        + (best.y - last_clicked.y) ** 2 <= 3 ** 2):
                    self.events.put(("log", "验证后仍选到上一步同一箭头，为保护生命已停止。"))
                    break
                arrow_bot.click_window(hwnd, offset[0] + best.x - best.dx * 5,
                                       offset[1] + best.y - best.dy * 5)
                last_clicked = best
                clicks += 1
                self.events.put(("log", f"已执行第 {clicks} 次点击"))
                if mode == "step":
                    # A single step still performs one follow-up recognition so
                    # the log can say whether the click actually removed it.
                    if not self.stop_event.wait(interval):
                        _h2, _o2, after, marked2, _t2, _r2 = self._recognize(title, threshold)
                        self.events.put(("preview", marked2))
                        repeated = bool(after and after[0].dx == best.dx and after[0].dy == best.dy
                                        and (after[0].x - best.x) ** 2
                                        + (after[0].y - best.y) ** 2 <= 3 ** 2)
                        self.events.put(("log", "验证失败：点击后同一候选仍存在。"
                                                if repeated else "验证通过：画面候选已变化。"))
                    break
                if self.stop_event.wait(interval):
                    break
        except Exception as exc:
            self.events.put(("error", f"{exc}\n\n{traceback.format_exc()}"))
        finally:
            self.events.put(("done", clicks))

    def _drain_events(self) -> None:
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "preview":
                    self._show_preview(data)  # type: ignore[arg-type]
                elif kind == "count":
                    self.count_var.set(f"候选：{data}")
                    self.status_var.set("识别完成" if data else "没有找到可靠候选")
                elif kind == "log":
                    self._append_log(str(data))
                elif kind == "status":
                    self.status_var.set(str(data))
                elif kind == "error":
                    self._append_log("错误：" + str(data))
                    self.status_var.set("运行出错，请查看日志")
                    messagebox.showerror("运行错误", str(data).split("\n\n", 1)[0])
                elif kind == "done":
                    self._set_running(False)
                    if self.stop_event.is_set():
                        self.status_var.set("已停止")
                    self._append_log(f"本轮结束，共点击 {data} 次。")
        except queue.Empty:
            pass
        self.after(80, self._drain_events)

    def _show_preview(self, bgr) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_w = max(400, self.preview.winfo_width() - 10)
        max_h = max(400, self.preview.winfo_height() - 10)
        image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.preview_photo, text="")

    def _append_log(self, line: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        with Path("arrow-bot.log").open("a", encoding="utf-8") as log_file:
            log_file.write(f"[{stamp}] {line}\n")
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, f"[{stamp}] {line}\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def open_debug(self) -> None:
        path = Path("debug-detection.png").resolve()
        if not path.exists():
            messagebox.showinfo("提示", "请先执行一次识别。")
            return
        import os
        os.startfile(path)

    def on_close(self) -> None:
        self.stop_event.set()
        self.destroy()


if __name__ == "__main__":
    ArrowBotGUI().mainloop()
