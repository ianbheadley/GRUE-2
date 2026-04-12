"""
Interactive label correction tool for SAM vehicle color labels.

Shows each CIFAR vehicle image with its SAM-assigned color label.
Click a color button to reassign, or press Next to accept and move on.
Corrections are saved to a separate JSON so the originals stay intact.

Usage (from GRUE root):
    python src/label_corrector.py
    python src/label_corrector.py --class_filter truck
    python src/label_corrector.py --color_filter red
    python src/label_corrector.py --class_filter truck --color_filter red
    python src/label_corrector.py --only_uncertain   # show only 'unknown' labels
"""

import argparse
import json
import os
import sys
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from cifar_dataset import load_raw_cifar10, CIFAR10_LABELS
from cifar_color_labels import load_vehicle_labels

COLORS = [
    "red", "orange", "yellow", "green", "blue",
    "purple", "pink", "brown", "grey", "black", "white", "unknown",
]

COLOR_HEX = {
    "red":     "#e53935",
    "orange":  "#fb8c00",
    "yellow":  "#c0a000",   # darkened so it's readable on white
    "green":   "#43a047",
    "blue":    "#1e88e5",
    "purple":  "#8e24aa",
    "pink":    "#e91e8c",
    "brown":   "#6d4c41",
    "grey":    "#757575",
    "black":   "#212121",
    "white":   "#9e9e9e",
    "unknown": "#aaaaaa",
}

CORRECTIONS_PATH = "dataset_cifar10/vehicle_color_corrections.json"


class LabelCorrectorApp:
    def __init__(self, root, indices, X_test, y_test, labels, corrections):
        self.root       = root
        self.indices    = indices          # list of CIFAR test indices to review
        self.X_test     = X_test
        self.y_test     = y_test
        self.labels     = labels           # original SAM labels
        self.corrections = corrections     # {str(idx): color} — user edits

        self.pos        = 0                # current position in indices
        self.n_changed  = sum(
            1 for k in corrections if int(k) in set(indices)
        )

        root.title("CIFAR Vehicle Label Corrector")
        root.configure(bg="#1e1e1e")
        root.resizable(False, False)
        root.bind("<Left>",  lambda e: self.prev())
        root.bind("<Right>", lambda e: self.next_image())
        root.bind("<Return>", lambda e: self.next_image())

        # ── Image panel ──────────────────────────────────────────────────────
        self.canvas = tk.Canvas(root, width=320, height=320, bg="#1e1e1e",
                                highlightthickness=0)
        self.canvas.pack(pady=(16, 0))

        # ── Info bar ─────────────────────────────────────────────────────────
        info_frame = tk.Frame(root, bg="#1e1e1e")
        info_frame.pack(fill="x", padx=16)

        self.lbl_pos   = tk.Label(info_frame, text="", bg="#1e1e1e",
                                  fg="#aaaaaa", font=("Helvetica", 11))
        self.lbl_pos.pack(side="left")

        self.lbl_changed = tk.Label(info_frame, text="", bg="#1e1e1e",
                                    fg="#43a047", font=("Helvetica", 11, "bold"))
        self.lbl_changed.pack(side="right")

        # ── Current label display ─────────────────────────────────────────────
        self.lbl_current = tk.Label(root, text="", bg="#1e1e1e",
                                    font=("Helvetica", 18, "bold"))
        self.lbl_current.pack(pady=(6, 2))

        self.lbl_class = tk.Label(root, text="", bg="#1e1e1e",
                                  fg="#cccccc", font=("Helvetica", 12))
        self.lbl_class.pack(pady=(0, 10))

        # ── Color buttons ────────────────────────────────────────────────────
        btn_frame = tk.Frame(root, bg="#1e1e1e")
        btn_frame.pack(padx=16, pady=4)

        self.color_buttons = {}
        cols = 6
        for i, color in enumerate(COLORS):
            hex_col = COLOR_HEX[color]
            # Use a canvas-based button: colored square + dark label text
            # so it's always readable regardless of macOS theming
            cell = tk.Frame(btn_frame, bg="#2a2a2a", cursor="hand2")
            cell.grid(row=i // cols, column=i % cols, padx=3, pady=3)

            swatch = tk.Canvas(cell, width=18, height=18, bg="#2a2a2a",
                               highlightthickness=0)
            swatch.pack(side="left", padx=(6, 2), pady=6)
            swatch.create_rectangle(2, 2, 16, 16, fill=hex_col, outline="#555555")

            lbl = tk.Label(cell, text=color, bg="#2a2a2a", fg="#ffffff",
                           font=("Helvetica", 10, "bold"), padx=4, pady=4)
            lbl.pack(side="left", padx=(0, 6))

            # Bind click on frame, canvas, and label
            for widget in (cell, swatch, lbl):
                widget.bind("<Button-1>", lambda e, c=color: self.assign_color(c))
                widget.bind("<Enter>",    lambda e, f=cell: f.config(bg="#3a3a3a"))
                widget.bind("<Leave>",    lambda e, f=cell: f.config(bg="#2a2a2a"))

            self.color_buttons[color] = cell

        # ── Nav buttons ──────────────────────────────────────────────────────
        nav_frame = tk.Frame(root, bg="#1e1e1e")
        nav_frame.pack(pady=10)

        tk.Button(nav_frame, text="◀  Prev", command=self.prev,
                  bg="#333333", fg="white", relief="flat",
                  font=("Helvetica", 11), width=10).pack(side="left", padx=6)

        tk.Button(nav_frame, text="Next  ▶", command=self.next_image,
                  bg="#333333", fg="white", relief="flat",
                  font=("Helvetica", 11), width=10).pack(side="left", padx=6)

        tk.Button(nav_frame, text="💾 Save", command=self.save,
                  bg="#1e88e5", fg="white", relief="flat",
                  font=("Helvetica", 11, "bold"), width=10).pack(side="left", padx=6)

        # ── Keyboard hint ─────────────────────────────────────────────────────
        tk.Label(root, text="← → arrow keys to navigate  |  Enter = next",
                 bg="#1e1e1e", fg="#555555", font=("Helvetica", 9)).pack(pady=(0, 10))

        self.render()

    # ── Rendering ────────────────────────────────────────────────────────────

    def current_idx(self):
        return self.indices[self.pos]

    def effective_color(self, idx):
        """Return the user-corrected color if set, else SAM label."""
        if str(idx) in self.corrections:
            return self.corrections[str(idx)]
        entry = self.labels.get(idx)
        return entry["color"] if entry and entry["color"] else "unknown"

    def render(self):
        idx   = self.current_idx()
        img32 = self.X_test[idx]
        color = self.effective_color(idx)
        cls   = CIFAR10_LABELS[int(self.y_test[idx])]
        corrected = str(idx) in self.corrections

        # Upscale image to 304×304
        img_uint8 = (img32 * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(img_uint8).resize((304, 304), Image.NEAREST)

        # Draw border in label color
        hex_col = COLOR_HEX.get(color, "#aaaaaa")
        r_val   = int(hex_col[1:3], 16)
        g_val   = int(hex_col[3:5], 16)
        b_val   = int(hex_col[5:7], 16)
        bordered = Image.new("RGB", (320, 320), (r_val, g_val, b_val))
        bordered.paste(pil, (8, 8))

        self._tk_img = ImageTk.PhotoImage(bordered)
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

        # Label text
        prefix = "✏️ " if corrected else ""
        self.lbl_current.config(
            text=f"{prefix}{color.upper()}",
            fg=hex_col if color != "white" else "#9e9e9e",
        )

        entry = self.labels.get(idx, {})
        iou   = entry.get("iou", 0) if entry else 0
        area  = entry.get("area_frac", 0) if entry else 0
        self.lbl_class.config(
            text=f"{cls}  ·  idx {idx}  ·  iou={iou:.2f}  area={area:.2f}"
        )

        self.lbl_pos.config(
            text=f"{self.pos + 1} / {len(self.indices)}"
        )
        self.lbl_changed.config(
            text=f"{self.n_changed} corrected"
        )

        # Highlight active color button with a bright border
        for c, frame in self.color_buttons.items():
            if c == color:
                frame.config(bg="#ffffff", highlightbackground="#ffffff",
                             highlightthickness=2)
                for child in frame.winfo_children():
                    child.config(bg="#ffffff")
                    if isinstance(child, tk.Label):
                        child.config(fg="#111111")
            else:
                frame.config(bg="#2a2a2a", highlightthickness=0)
                for child in frame.winfo_children():
                    child.config(bg="#2a2a2a")
                    if isinstance(child, tk.Label):
                        child.config(fg="#ffffff")

    # ── Actions ──────────────────────────────────────────────────────────────

    def assign_color(self, color):
        idx = self.current_idx()
        was_corrected = str(idx) in self.corrections
        original_color = (self.labels.get(idx) or {}).get("color") or "unknown"

        if color == original_color and was_corrected:
            # User reverted to original — remove correction
            del self.corrections[str(idx)]
            self.n_changed -= 1
        elif color != original_color or not was_corrected:
            if not was_corrected:
                self.n_changed += 1
            self.corrections[str(idx)] = color

        self.render()
        self.next_image()   # auto-advance after labeling

    def next_image(self):
        if self.pos < len(self.indices) - 1:
            self.pos += 1
            self.render()

    def prev(self):
        if self.pos > 0:
            self.pos -= 1
            self.render()

    def save(self):
        os.makedirs(os.path.dirname(CORRECTIONS_PATH), exist_ok=True)
        with open(CORRECTIONS_PATH, "w") as f:
            json.dump(self.corrections, f, indent=2)
        self.lbl_changed.config(text=f"{self.n_changed} corrected — SAVED ✓")
        print(f"Saved {len(self.corrections)} corrections → {CORRECTIONS_PATH}")


def load_corrections():
    if os.path.exists(CORRECTIONS_PATH):
        with open(CORRECTIONS_PATH) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Interactive vehicle label corrector")
    parser.add_argument("--labels_path", default="dataset_cifar10/vehicle_color_labels.json")
    parser.add_argument("--class_filter", default=None, choices=["airplane", "automobile", "ship", "truck"],
                        help="Only show images of this CIFAR class")
    parser.add_argument("--color_filter", default=None,
                        help="Only show images with this SAM color label")
    parser.add_argument("--only_uncertain", action="store_true",
                        help="Only show images where SAM returned 'unknown'")
    parser.add_argument("--shuffle", action="store_true", default=True,
                        help="Shuffle order (default: on)")
    args = parser.parse_args()

    print("Loading CIFAR-10 test set...")
    _, _, X_test, y_test = load_raw_cifar10()

    print(f"Loading labels from {args.labels_path}...")
    labels = load_vehicle_labels(args.labels_path)
    corrections = load_corrections()
    if corrections:
        print(f"Loaded {len(corrections)} existing corrections from {CORRECTIONS_PATH}")

    # Build filtered index list
    VEHICLE_IDS = {0, 1, 8, 9}
    indices = []
    for idx, entry in labels.items():
        cifar_class = CIFAR10_LABELS[int(y_test[idx])]
        color = entry["color"] or "unknown"

        if args.class_filter and cifar_class != args.class_filter:
            continue
        if args.only_uncertain and color != "unknown":
            continue
        if args.color_filter and color != args.color_filter:
            continue
        indices.append(idx)

    if not indices:
        print("No images match the filter criteria.")
        sys.exit(1)

    if args.shuffle:
        np.random.seed(0)
        np.random.shuffle(indices)

    print(f"Showing {len(indices)} images.")

    root = tk.Tk()
    app = LabelCorrectorApp(root, indices, X_test, y_test, labels, corrections)

    def on_close():
        app.save()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
