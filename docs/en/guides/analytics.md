# Analytics using VajraV1

## Introduction

This guide provides a comprehensive overview of three fundamental types of data visualizations: line graphs, bar plots, and pie charts. Each section includes step-by-step instructions and code snippets on how to create these visualizations using Python.

## The Importance of Graphs in Analytics

- Line graphs are ideal for tracking changes over short and long periods and for comparing changes for multiple groups over the same period.
- Bar plots, on the other hand, are suitable for comparing quantities across different categories and showing relationships between a category and its numerical value.
- Lastly, pie charts are effective for illustrating proportions among categories and showing parts of a whole.

!!! example "Analytics using Vayuvahana Technologies VayuAI's VajraV1"

    === "Python"

        ```python
        import cv2
        from vajra import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(
            "analytics_output.avi",
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (1280, 720),
        )

        analytics = solutions.Analytics(
            show=True,
            analytics_type="line"
            model="vajra-v1-nano-det.pt"
        )

        frame_count = 0
        while cap.isOpened():
            success, im0 = cap.read()
            if success:
                frame_count += 1
                results = analytics(im0, frame_count)
                out.write(results.plot_im)
            else:
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        ```
    
    === "CLI"

        ```bash
        vajra solutions analytics show=True

        # Pass the video
        vajra solutions analytics source="path/to/video.mp4"

        # Generate pie plot
        vajra solutions analytics analytics_type="pie" show=True

        # Generate bar plot
        vajra solutions analytics analytics_type="bar" show=True
        
        # Generat area plot
        vajra solutions analytics analytics_type="area" show=True
        ```

## `Analytics` Arguments

Here's a table outlining the Analytics arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "analytics_type"]) }}

You can also leverage different [`track`](../modes/track.md) arguments in the `Analytics` solution.

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Additionally, the following visualization arguments are supported:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width"]) }}

## Conclusion

It is important to understand when and how to use different types of visualizations is crucial for effective data analysis. Line graphs, bar plots, and pie charts are fundamental tools that can help you convey your data's story more clearly and effectively. Vayuvahana Technologies VayuAI's Analytics Solution provides a streamlined way to generate these visualizations from your object detection and tracking results, making it easier to extract meaningful insights from your visual data.

## FAQ

### How do I create a line graph?

To create a line graph, you can utilise the Analytics Solution offered by the VayuAI SDK. Follow these steps:

1. Load a VajraV1 model and open your video file.
2. Initiate the Analytics class with the type set to "line".
3. Iterate through video frames, updating the line graph with relevant data, such as object counts per frame.
4. Save the output video displaying the line graph.

Example:

```python
import cv2
from vajra import solutions
cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "vayuvahana_analytics.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (1280, 720),  # this is fixed
)

analytics = solutions.Analytics(
    analytics_type="line",
    show=True,
)

frame_count = 0

while cap.isOpened():
    success, im0 = cap.read()
    if success:
        frame_count += 1
        results = analytics(im0, frame_count)
        out.write(results.plot_im)
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

For further details look at the [Analytics using VajraV1](#analytics-using-vajrav1) section.

### What are the benefits of using VajraV1 for creating bar plots?

Using VajraV1 for creating bar plots offers several benefits:

1. **Real-time Data Visualization**: Seamlessly integrate object detection results into bar plots for dynamic updates.
2. **Ease of Use**: Simple API and functions make it straightforward to implement and visualize data.
3. **Customization**: Customize titles, labels, colors, and more to fit your specific requirements.
4. **Efficiency**: Efficiently handle large amounts of data and update plots in real-time during video processing.

```python
import cv2

from vajra import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "vayuvahana_analytics.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (1280, 720),  # this is fixed
)

analytics = solutions.Analytics(
    analytics_type="bar",
    show=True,
)

frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if success:
        frame_count += 1
        results = analytics(im0, frame_count)
        out.write(results.plot_im)
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```
