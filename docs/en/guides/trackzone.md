# TrackZone using Vayuvahana Technologies VayuAI

## What is TrackZone?

TrackZone specializes in monitoring objects within designated areas of a frame instead of the whole frame. Built on [Vayuvahana Technologies VayuAI](https://github.com/NamanMakkar/VayuAI), it integrates object detection and tracking specifically within zones for videos and live camera feeds. VajraV1's advanced deep learning models make it a perfect choice for real-time use cases, offering precise and efficient object tracking in applications like crowd monitoring and surveillance.

## Advantages of Object Tracking in Zones (TrackZone)

- **Targeted Analysis:** Tracking objects within specific zones allows for more focused insights, enabling precise monitoring and analysis of areas of interest, such as entry points or restricted zones.
- **Improved Efficiency:** By narrowing the tracking scope to defined zones, TrackZone reduces computational overhead, ensuring faster processing and optimal performance.
- **Enhanced Security:** Zonal tracking improves surveillance by monitoring critical areas, aiding in the early detection of unusual activity or security breaches.
- **Scalable Solutions:** The ability to focus on specific zones makes TrackZone adaptable to various scenarios, from retail spaces to industrial settings, ensuring seamless integration and scalability.

## Usage Examples

!!! example "TrackZone using Vayuvahana Technologies VajraV1"

    === "CLI"

        ```bash
        # Run a trackzone example
        vajra solutions trackzone show=True

        # Pass a source video
        vajra solutions trackzone show=True source="path/to/video.mp4"

        # Pass region coordinates
        vajra solutions trackzone show=True region="[(150, 150), (1130, 150), (1130, 570), (150, 570)]"
        ```

    === "Python"

        ```python
        import cv2

        from vajra import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Define region points
        region_points = [(150, 150), (1130, 150), (1130, 570), (150, 570)]

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("trackzone_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init trackzone (object tracking in zones, not complete frame)
        trackzone = solutions.TrackZone(
            show=True,  # display the output
            region=region_points,  # pass region points
            model="vajra-v1-nano-det.pt", # use any of the supported detection models
            # line_width=2,  # adjust the line width for bounding boxes and text display
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = trackzone(im0)

            # print(results)  # access the output

            video_writer.write(results.plot_im)  # write the video file

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

## `TrackZone` Arguments

Here's a table with the `TrackZone` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "region"]) }}

The TrackZone solution includes support for `track` parameters:

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Moreover, the following visualization options are available:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## FAQ

### How do I track objects in a specific area or zone of a video frame using Vayuvahana Technologies VajraV1?

Tracking objects in a defined area or zone of a video frame is straightforward with Vayuvahana Technologies VajraV1. Simply use the command provided below to initiate tracking. This approach ensures efficient analysis and accurate results, making it ideal for applications like surveillance, crowd management or any scenario requiring zonal tracking.

```bash
vajra solutions trackzone source="path/to/video.mp4" show=True
```

### How can I use TrackZone in Python with Vayuvahana Technologies VayuAI?

You can set up object tracking in specific zones, making it easy to integrate it into your project using the code below:

```python
import cv2

from vajra import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points
region_points = [(150, 150), (1130, 150), (1130, 570), (150, 570)]

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init trackzone (object tracking in zones, not complete frame)
trackzone = solutions.TrackZone(
    show=True,  # display the output
    region=region_points,  # pass region points
    model="vajra-v1-nano-det.pt",
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    results = trackzone(im0)
    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```

### How do I configure the zone points for video processing using VayuAI's TrackZone?

Configuring zone points for video processing is simple. You can directly define and adjust the zones through a Python script, allowing precise control over the areas you want to monitor.

```python
# Define region points
region_points = [(150, 150), (1130, 150), (1130, 570), (150, 570)]

# Initialize trackzone
trackzone = solutions.TrackZone(
    show=True,  # display the output
    region=region_points,  # pass region points
)
```
