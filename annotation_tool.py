"""Small point-annotation GUI for arrow heads and directions."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from dataset_pipeline import (CLASSES, RAW_IMAGES, annotation_path, ensure_directories,
                              load_annotations, save_annotations)


KEY_TO_CLASS = {
    "1": 0, "2": 1, "3": 2, "4": 3,
    "5": 4, "6": 5, "7": 6, "8": 7,
    "u": 0, "r": 1, "d": 2, "l": 3,
}


class AnnotationTool(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        ensure_directories()
        self.title("一箭又一箭 - 箭头标注工具")
        self.geometry("1120x820")
        self.images = sorted(RAW_IMAGES.glob("*.png"))
        self.index = 0
        self.class_id = tk.IntVar(value=0)
        self.objects: list[dict] = []
        self.source: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.scale = 1.0
        self.offset = (0, 0)
        self.status = tk.StringVar()
        self._build()
        self.bind("<Key>", self.on_key)
        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<Button-3>", self.delete_nearest)
        self.canvas.bind("<Configure>", lambda _e: self.render())
        self.update_idletasks()
        if self.images:
            self.load_current()
        else:
            self.status.set("没有采集图片，请先在主工具中点击“采集训练截图”")

    def _build(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)
        for idx, name in enumerate(CLASSES):
            ttk.Radiobutton(top, text=f"{idx + 1} {name}", variable=self.class_id,
                            value=idx).pack(side=tk.LEFT, padx=3)
        nav = ttk.Frame(self, padding=(8, 0, 8, 6))
        nav.pack(fill=tk.X)
        ttk.Button(nav, text="上一张", command=self.previous).pack(side=tk.LEFT)
        ttk.Button(nav, text="撤销/右键", command=self.undo).pack(side=tk.LEFT, padx=6)
        ttk.Button(nav, text="保存", command=self.save).pack(side=tk.LEFT, padx=6)
        ttk.Button(nav, text="保存并下一张", command=self.save_next).pack(side=tk.LEFT, padx=6)
        ttk.Label(nav, textvariable=self.status).pack(side=tk.RIGHT)
        self.canvas = tk.Canvas(self, background="#222", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    def load_current(self) -> None:
        path = self.images[self.index]
        self.source = Image.open(path).convert("RGB")
        self.objects = load_annotations(path)
        self.render()
        self._status()

    def _status(self) -> None:
        if not self.images:
            return
        done = sum(annotation_path(p).exists() for p in self.images)
        self.status.set(f"{self.index + 1}/{len(self.images)} | 当前 {len(self.objects)} 个 | 已标注 {done} 张")

    def render(self) -> None:
        if self.source is None:
            return
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        self.scale = min(width / self.source.width, height / self.source.height)
        draw_w = max(1, int(self.source.width * self.scale))
        draw_h = max(1, int(self.source.height * self.scale))
        self.offset = ((width - draw_w) // 2, (height - draw_h) // 2)
        resized = self.source.resize((draw_w, draw_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(self.offset[0], self.offset[1], anchor=tk.NW, image=self.photo)
        colors = ["#00ff66"] * 4 + ["#ff3bff"] * 4
        arrows = ["↑", "→", "↓", "←"]
        for obj in self.objects:
            x = self.offset[0] + obj["x"] * self.scale
            y = self.offset[1] + obj["y"] * self.scale
            cid = int(obj["class_id"])
            self.canvas.create_oval(x - 8, y - 8, x + 8, y + 8,
                                    outline=colors[cid], width=2)
            self.canvas.create_text(x + 12, y - 12, text=arrows[cid % 4],
                                    fill=colors[cid], font=("Segoe UI", 15, "bold"))

    def add_point(self, event) -> None:
        if self.source is None:
            return
        x = (event.x - self.offset[0]) / self.scale
        y = (event.y - self.offset[1]) / self.scale
        if 0 <= x < self.source.width and 0 <= y < self.source.height:
            self.objects.append({"x": round(x, 2), "y": round(y, 2),
                                 "class_id": int(self.class_id.get())})
            self.render()
            self._status()

    def undo(self) -> None:
        if self.objects:
            self.objects.pop()
            self.render()
            self._status()

    def delete_nearest(self, event) -> None:
        if not self.objects:
            return
        x = (event.x - self.offset[0]) / self.scale
        y = (event.y - self.offset[1]) / self.scale
        nearest = min(range(len(self.objects)),
                      key=lambda i: (self.objects[i]["x"] - x) ** 2
                      + (self.objects[i]["y"] - y) ** 2)
        distance = ((self.objects[nearest]["x"] - x) ** 2
                    + (self.objects[nearest]["y"] - y) ** 2) ** 0.5
        if distance <= 25 / self.scale:
            self.objects.pop(nearest)
            self.render()
            self._status()

    def save(self) -> None:
        if self.images:
            save_annotations(self.images[self.index], self.objects)
            self._status()

    def save_next(self) -> None:
        self.save()
        if self.index + 1 < len(self.images):
            self.index += 1
            self.load_current()

    def previous(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.load_current()

    def on_key(self, event) -> None:
        key = event.char.lower()
        if key in KEY_TO_CLASS:
            cid = KEY_TO_CLASS[key]
            if event.state & 0x0001 and cid < 4:
                cid += 4
            self.class_id.set(cid)
        elif event.keysym == "BackSpace":
            self.undo()
        elif event.keysym in ("Return", "space"):
            self.save_next()


if __name__ == "__main__":
    AnnotationTool().mainloop()
