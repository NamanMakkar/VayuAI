# Heatmaps using VajraV1

## Introduction

A heatmap generated with the VajraV1 models uses a spectrum of colors to represent varying data values, warmer hues indicating higher intensities and cooler signifying lower values. Heatmaps excel in visualizing intricate data patterns, correlations and anomalies, offering an accessible and engaging approach to data interpretation across diverse domains.

## Why Choose Heatmaps for Data Analysis?

- **Intuitive Data Distribution Visualization:** Heatmaps simplify the comprehension of data concentration and distribution, converting complex datasets into easy-to-understand visual formats.
- **Efficient Pattern Detection:** By visualizing data in heatmap format, it becomes easier to spot trends, clusters, and outliers, facilitating quicker analysis and insights.
- **Enhanced Spatial Analysis and Decision-Making:** Heatmaps are instrumental in illustrating spatial relationships, aiding in decision-making processes in sectors such as business intelligence, environmental studies, and urban planning.

## Usage Example

!!! example "Heatmap using VajraV1"

    === "Python"

        ```python
        import cv2
        from vajra import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        video_writer = cv2.VideoWriter("heatmap_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        heatmap_obj = solutions.Heatmap(
            colormap=cv2.COLORMAP_PARULA,  # Color of the heatmap
            show=True,  # Display the image during processing
            model="vajra-v1-nano-det.pt", 
        )

        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = heatmap_obj(im0)

            print(results)

            video_writer.write(results.plot_im)
        
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```
    
    === "CLI"

        ```bash
        # Pass a source video
        vajra solutions heatmap source="path/to/video.mp4"

        # Heatmap with custom colormap
        vajra solutions heatmap colormap=cv2.COLORMAP_INFERNO

        # Heatmaps + region object counting
        vajra solutions heatmap region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]"
        ```

### `Heatmap()` Arguments

Here's a table with the `Heatmap` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "colormap", "show_in", "show_out", "region"]) }}

You can also apply different `track` arguments in the `Heatmap` solution.

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Additionally, the supported visualization arguments are listed below:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

### Heatmap COLORMAPs

| Colormap Name                   | Description                            |
| ------------------------------- | -------------------------------------- |
| `cv::COLORMAP_AUTUMN`           | Autumn color map                       |
| `cv::COLORMAP_BONE`             | Bone color map                         |
| `cv::COLORMAP_JET`              | Jet color map                          |
| `cv::COLORMAP_WINTER`           | Winter color map                       |
| `cv::COLORMAP_RAINBOW`          | Rainbow color map                      |
| `cv::COLORMAP_OCEAN`            | Ocean color map                        |
| `cv::COLORMAP_SUMMER`           | Summer color map                       |
| `cv::COLORMAP_SPRING`           | Spring color map                       |
| `cv::COLORMAP_COOL`             | Cool color map                         |
| `cv::COLORMAP_HSV`              | HSV (Hue, Saturation, Value) color map |
| `cv::COLORMAP_PINK`             | Pink color map                         |
| `cv::COLORMAP_HOT`              | Hot color map                          |
| `cv::COLORMAP_PARULA`           | Parula color map                       |
| `cv::COLORMAP_MAGMA`            | Magma color map                        |
| `cv::COLORMAP_INFERNO`          | Inferno color map                      |
| `cv::COLORMAP_PLASMA`           | Plasma color map                       |
| `cv::COLORMAP_VIRIDIS`          | Viridis color map                      |
| `cv::COLORMAP_CIVIDIS`          | Cividis color map                      |
| `cv::COLORMAP_TWILIGHT`         | Twilight color map                     |
| `cv::COLORMAP_TWILIGHT_SHIFTED` | Shifted Twilight color map             |
| `cv::COLORMAP_TURBO`            | Turbo color map                        |
| `cv::COLORMAP_DEEPGREEN`        | Deep Green color map                   |

These colormaps are commonly used for visualizing data with different color representations.

## How Heatmaps Work in Vayuvahana Technologies VayuAI

The Heatmap solution extends the ObjectCounter class to generate and visualize movement patterns in video streams. When initialized, the solution creates a blank heatmap layer that gets updated as objects move through the frame.

1. Tracks the object across frames using VajraV1's tracking capabilities
2. Updates the heatmap intensity at the object's location
3. Applies a selected colormap to visualize the intensity values
4. Overlays the colored heatmap on the original frame

The result is a dynamic visualization that builds up over time, revealing traffic patterns, crowd movements, or other spatial behaviors in your video data.

## FAQ

### Can I use the VajraV1 to perform object tracking and generate a heatmap simultaneously?

Yes VajraV1 supports object tracking and heatmap generation concurrently. This can be achieved through the Heatmap solution integrated with object tracking models. To do so, you need to initialize the heatmap object and use VajraV1's tracking capabilities. Here is an example:

```python
import cv2

from vajra import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
heatmap = solutions.Heatmap(colormap=cv2.COLORMAP_PARULA, show=True, model="vajra-v1-nano-det.pt")

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    results = heatmap(im0)

cap.release()
cv2.destroyAllWindows()
```

For further guidance, check the [Tracking Mode](../modes/track.md) page.