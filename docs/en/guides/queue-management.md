# Queue Management with VajraV1

## What is Queue Management?

Queue management using VajraV1 involves organizing and controlling lines of people or vehicles to resuce wait times and enhance efficiency. It helps optimize queues to improve customer satisfaction and system performance in various settings like retail, banks, airports and healthcare facilities.

## Advantages of Queue Management

- **Reduced Waiting Times:** Queue management systems efficiently organize queues, minimizing wait times for customers. This leads to improved satisfaction levels as customers spend less time waiting and more time engaging with products or services.
- **Increased Efficiency:** Implementing queue management allows businesses to allocate resources more effectively. By analyzing queue data and optimizing staff deployment, businesses can streamline operations, reduce costs, and improve overall productivity.
- **Real-time Insights:** VajraV1 powered queue management provides instant data on queue lengths and wait times, enabling managers to make informed decisions quickly.
- **Enhanced Customer Experience:** By reducing frustration associated with long waits, businesses can significantly improve customer satisfaction and loyalty.

## Usage Example

!!! example "Queue Management using Vayuvahana Technologies VayuAI's VajraV1"

    === "Python"

        ```python
        import cv2
        from vajra import solutions

        cap = cv2.ideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("queue_management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Define queue points
        queue_region = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

        queuemanager = solutions.QueueManager(
            show=True,
            model="vajra-v1-nano-det.pt",
            region=queue_region,
        )

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or processing is complete.")
                break
            results = queuemanager(im0)


            video_writer.write(results.plot_im)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```
    
    === "CLI"

        ```bash
        vajra solutions queue show=True

        vajra solutions queue source="path/to/video.mp4"

        vajra solutions queue region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]"
        ```

## `QueueManager` Arguments

Here's a table with the `QueueManager` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "region"]) }}

The `QueueManagement` solution also support some `track` arguments:

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Additionally, the following visualization parameters are available:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## Implementation Strategies

When implementing queue management with VajraV1, consider these best practices:

1. **Strategic Camera Placement:** Position cameras to capture the entire queue area without obstructions.
2. **Define Appropriate Queue Regions:** Carefully set queue boundaries based on the physical layout of your space.
3. **Adjust Detection Confidence:** Fine-tune the confidence threshold based on lighting conditions and crowd density.
4. **Integrate with Existing Systems:** Connect your queue management solution with digital signage or staff notification systems for automated responses.

## FAQ

### How can I use VajraV1 for real-time queue management?

Follow these steps:

1. Load a VajraV1 model.
2. Capture the video feed using cv2.VideoCapture.
3. Define the region of interest (ROI) for queue management.
4. Process frames to detect objects and manage queues.

Minimal Example:

```python
import cv2

from vajra import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
queue_region = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

queuemanager = solutions.QueueManager(
    model="vajra-v1-nano-det.pt",
    region=queue_region,
    line_width=3,
    show=True
)

while cap.isOpened():
    success, im0 = cap.read()
    if success:
        results = queuemanager(im0)

cap.release()
cv2.destroyAllWindows()
```