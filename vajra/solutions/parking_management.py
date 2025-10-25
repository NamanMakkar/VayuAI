import json
from typing import Any

import cv2
import numpy as np

from vajra.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from vajra.utils import LOGGER
from vajra.checks import check_imshow

class ParkingPtsSelection:
    def __init__(self) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox
        except ImportError:
            import platform

            install_cmd = {
                "Linux": "sudo apt install python3-tk (Debian/Ubuntu) | sudo dnf install python3-tkinter (Fedora) | "
                "sudo pacman -S tk (Arch)",
                "Windows": "reinstall Python and enable the checkbox `tcl/tk and IDLE` on **Optional Features** during installation",
                "Darwin": "reinstall Python from https://www.python.org/downloads/macos/ or `brew install python-tk`",
            }.get(platform.system(), "Unknown OS. Check your Python installation.")

            LOGGER.warning(f" Tkinter is not configured or supported. Potential fix: {install_cmd}")
            return
        
        if not check_imshow(warn=True):
            return
        
        self.tk, self.filedialog, self.messagebox = tk, filedialog, messagebox
        self.master = self.tk.Tk()
        self.master.title("Vayuvahana Technologies Parking Zones Point Selector")
        self.master.resizable(False, False)

        self.canvas = self.tk.Canvas(self.master, bg="white")
        self.canvas.pack(side=self.tk.BOTTOM)

        self.image = None
        self.canvas_image = None
        self.canvas_max_width = None
        self.canvas_max_height = None
        self.rg_data = None
        self.current_box = None
        self.imgh = None
        self.imgw = None

        button_frame = self.tk.Frame(self.master)
        button_frame.pack(side=self.tk.TOP)

        for text, cmd in [
            ("Upload Image", self.upload_image),
            ("Remove Last BBox", self.remove_last_bounding_box),
            ("Save", self.save_to_json),
        ]:
            self.tk.Button(button_frame, text=text, command=cmd).pack(side=self.tk.LEFT)
        
        self.initialize_properties()
        self.master.mainloop()

    def initialize_properties(self) -> None:
        self.image = self.canvas_image = None
        self.rg_data, self.current_box = [], []
        self.imgw = self.imgh = 0
        self.canvas_max_width, self.canvas_max_height = 1280, 720

    def upload_image(self) -> None:
        from PIL import Image, ImageTk

        file = self.filedialog.askopenfilename(filetypes=[("Image Files", "*.png", "*.jpg", "*.jpeg")])
        if not file:
            LOGGER.info("No image selected.")
            return

        self.image = Image.open(file)
        self.imgw, self.imgh = self.image.size
        aspect_ratio = self.imgw / self.imgh
        canvas_width = (
            min(self.canvas_max_width, self.imgw) if aspect_ratio > 1 else int(self.canvas_max_height * aspect_ratio)
        )
        canvas_height = (
            min(self.canvas_max_height, self.imgh) if aspect_ratio <= 1 else int(self.canvas_max_width / aspect_ratio)
        )

        self.canvas.config(width=canvas_width, height=canvas_height)
        self.canvas_image = ImageTk.PhotoImage(self.image.resize((canvas_width, canvas_height)))
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.rg_data.clear(), self.current_box.clear()

    def on_canvas_click(self, event) -> None:
        self.current_box.append((event.x, event.y))
        self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill="red")
        if len(self.current_box) == 4:
            self.rg_data.append(self.current_box.copy())
            self.draw_box(self.current_box)
            self.current_box.clear()

    def draw_box(self, box: list[tuple[int, int]]) -> None:
        for i in range(4):
            self.canvas.create_line(box[i], box[(i + 1) % 4], fill="blue", width=2)

    def remove_last_bounding_box(self) -> None:
        if not self.rg_data:
            self.messagebox.showwarning("Warning", "No bounding boxes to remove.")
            return
        self.rg_data.pop()
        self.redraw_canvas()

    def redraw_canvas(self) -> None:
        self.canvas.delete["all"]
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)
        for box in self.rg_data:
            self.draw_box(box)

    def save_to_json(self) -> None:
        scale_w, scale_h = self.imgw / self.canvas.winfo_width(), self.imgh / self.canvas.winfo_height()
        data = [{"points": [(int(x * scale_w), int(y * scale_h)) for x, y in box]} for box in self.rg_data]

        from io import StringIO

        write_buffer = StringIO()
        json.dump(data, write_buffer, indent=4)
        with open("bounding_boxes.json", "w", encoding="utf-8") as f:
            f.write(write_buffer.getvalue())

        self.messagebox.showinfo("Success", "Bounding boxes saved to bounding_boxes.json")

class ParkingManagement(BaseSolution):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.json_file = self.CFG["json_file"]

        if self.json_file is None:
            LOGGER.warning("json_file argument is missing. Parking region details required.")
            raise ValueError("Json file path cannot be empty")
        
        with open(self.json_file) as f:
            self.json = json.load(f)

        self.pr_info = {"Occupancy": 0, "Available": 0}

        self.arc = (0, 0, 255)
        self.occ = (0, 255, 0)
        self.dc = (255, 0, 189)

    def process(self, im0: np.ndarray) -> SolutionResults:
        self.extract_tracks(im0)
        es, fs = len(self.json), 0 # Empty slots, filled slots
        annotator = SolutionAnnotator(im0, self.line_width)

        for region in self.json:
            pts_array = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
            rg_occupied = False
            for box, cls in zip(self.boxes, self.clss):
                xc, yc = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                dist = cv2.pointPolygonTest(pts_array, (xc, yc), False)

                if dist >=0:
                    annotator.display_objects_labels(
                        im0, self.model.names[int(cls)], (104, 31, 17), (255, 255, 255), xc, yc, 10
                    )
                    rg_occupied = True
                    break
            fs, es = (fs + 1, es - 1) if rg_occupied else (fs, es)
            cv2.ploylines(im0, [pts_array], isClosed=True, color=self.occ if rg_occupied else self.arc, thickness=2)
        
        self.pr_info["Occupancy"], self.pr_info["Available"] = fs, es
        annotator.display_analytics(im0, self.pr_info, (104, 31, 17), (255, 255, 255), 10)
        plot_im = annotator.result()
        self.display_output(plot_im)

        return SolutionResults(
            plot_im=plot_im,
            filled_slots=self.pr_info["Occupancy"],
            available_slots=self.pr_info["Available"],
            total_tracks=len(self.track_ids),
        )