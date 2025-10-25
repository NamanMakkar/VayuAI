from itertools import cycle
from typing import Any

import cv2
import numpy as np

from vajra.solutions.solutions import BaseSolution, SolutionResults

class Analytics(BaseSolution):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.type = self.CFG["analytics_type"]
        self.x_label = "Classes" if self.type in {"bar", "pie"} else "Frame#"
        self.y_label = "Total Counts"

        self.bg_color = "#F3F3F3"
        self.fg_color = "#111E68"
        self.title = "Vayuvahana Technologies Solutions"
        self.max_points = 45
        self.fontsize = 25
        figsize = self.CFG["figsize"]
        self.color_cycle = cycle(["#DD00BA", "#042AFF", "#FF4447", "#7D24FF", "#BD00FF"])

        self.total_counts = 0
        self.clswise_count = {}
        self.update_every = kwargs.get("update_every", 30)
        self.last_plot_im = None

        if self.type in {"line", "area"}:
            self.lines = {}
            self.fig = Figure(facecolor=self.bg_color, figsize=figsize)
            self.canvas = FigureCanvasAgg(self.fig)
            self.ax = self.fig.add_subplot(111, facecolor=self.bg_color)
            if self.type == "line":
                (self.line,) = self.ax.plot([], [], color="cyan", linewidth=self.line_width)
        elif self.type in {"bar", "pie"}:
            self.fig, self.ax = plt.subplots(figsize=figsize, facecolor=self.bg_color)
            self.canvas = FigureCanvasAgg(self.fig)
            self.ax.set_facecolor(self.bg_color)
            self.color_mapping = {}

            if self.type == "pie":
                self.ax.axis("equal")
    
    def process(self, im0: np.ndarray, frame_number: int) -> SolutionResults:
        self.extract_tracks(im0)
        if self.type == "line":
            for _ in self.boxes:
                self.total_counts += 1
            update_required = frame_number % self.update_every == 0 or self.last_plot_im is None
            if update_required:
                self.last_plot_im = self.update_graph(frame_number=frame_number)
            plot_im = self.last_plot_im
            self.total_counts = 0
        elif self.type in {"pie", "bar", "area"}:
            from collections import Counter

            self.clswise_count = Counter(self.names[int(cls)] for cls in self.clss)
            update_required = frame_number % self.update_every == 0 or self.last_plot_im is None
            if update_required: 
                self.last_plot_im = self.update_graph(
                    frame_number=frame_number, count_dict=self.clswise_count, plot=self.type
                )
            plot_im = self.last_plot_im
        else:
            raise ModuleNotFoundError(f"{self.type} chart is not supported")
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids), classwise_count=self.clswise_count)
    
    def update_graph(self, frame_number: int, count_dict: dict[str, int] | None = None, plot: str = "line") -> np.ndarray:
        if count_dict is None:
            x_data = np.append(self.line.get_xdata(), float(frame_number))
            y_data = np.append(self.line.get_ydata(), float(self.total_counts))

            if len(x_data) > self.max_points:
                x_data, y_data = x_data[-self.max_points :], y_data[-self.max_points :]
            
            self.line.set_data(x_data, y_data)
            self.line.set_label("Counts")
            self.line.set_marker("*")
            self.line.set_markersize(self.line_width * 5)
        
        else:
            labels = list(count_dict.keys())
            counts = list(count_dict.values())
            if plot == "area":
                color_cycle = cycle(["#DD00BA", "#042AFF", "#FF4447", "#7D24FF", "#BD00FF"])

                x_data = self.ax.lines[0].get_xdata() if self.ax.lines else np.array([])
                y_data_dict = {key: np.array([]) for key in count_dict.keys()}
                if self.ax.lines:
                    for line, key in zip(self.ax.lines, count_dict.keys()):
                        y_data_dict[key] = line.get_ydata()
                
                x_data = np.append(x_data, float(frame_number))
                max_length = len(x_data)

                for key in count_dict.keys():
                    y_data_dict[key] = np.append(y_data_dict[key], float(count_dict[key]))
                    if len(y_data_dict[key]) < max_length:
                        y_data_dict[key] = np.pad(y_data_dict[key], (0, max_length - len(y_data_dict[key])))
                if len(x_data) > self.max_points:
                    x_data = x_data[1:]
                    for key in count_dict.keys():
                        y_data_dict[key] = y_data_dict[key][1:]

                self.ax.clear()
                for key, y_data in y_data_dict.items():
                    color = next(color_cycle)
                    self.ax.fill_between(x_data, y_data, color=color, alpha=0.55)
                    self.ax.plot(
                        x_data,
                        y_data,
                        color=color,
                        linewidth=self.line_width,
                        marker="o",
                        markersize=self.line_width * 5,
                        label=f"{key} Data Points",
                    )
            elif plot == "bar":
                self.ax.clear()
                for label in labels:
                    if label not in self.color_mapping:
                        self.color_mapping[label] = next(self.color_cycle)
                colors = [self.color_mapping[label] for label in labels]
                bars = self.ax.bar(labels, counts, color=colors)
                for bar, count in zip(bars, counts):
                    self.ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        str(count),
                        ha="center",
                        va="bottom",
                        color=self.fg_color,
                    )
                
                for bar, label in zip(bars, labels):
                    bar.set_label(label)
                self.ax.legend(loc="upper left", fontsize=13, facecolor=self.fg_color, edgecolor=self.fg_color)
            elif plot == "pie":
                total = sum(counts)
                percentages = [size / total * 100 for size in counts]
                self.ax.clear()

                start_angle = 90

                wedges, _ = self.ax.pie(
                    counts, labels=labels, startangle=start_angle, textprops={"color": self.fg_color}, autopct=None
                )
                legend_labels = [f"{label} ({percentage:.1f}%)" for label, percentage in zip(labels, percentages)]
                self.ax.legend(wedges, legend_labels, title="Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                self.fig.subplots_adjust(left=0.1, right=0.75)

        self.ax.set_facecolor("#f0f0f0")
        self.ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        self.ax.set_title(self.title, color=self.fg_color, fontsize=self.fontsize)
        self.ax.set_xlabel(self.x_label, color=self.fg_color, fontsize=self.fontsize - 3)
        self.ax.set_ylabel(self.y_label, color=self.fg_color, fontsize=self.fontsize - 3)

        legend = self.ax.legend(loc="upper left", fontsize=13, facecolor=self.bg_color, edge_color=self.bg_color)
        for text in legend.get_texts():
            text.set_color(self.fg_color)

        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
        im0 = np.array(self.canvas.renderer.buffer_rgba())
        im0 = cv2.cvtColor(im0[:, :, :3], cv2.COLOR_RGBA2BGR)
        self.display_output(im0)

        return im0