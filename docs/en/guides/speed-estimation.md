# Speed Estimation using Vayuvahana Technologies VayuAI's VajraV1

## What is Speed Estimation?

Speed Estimation is the process of calculating the rate of movement of an object within a given context, often employed in computer vision applications. Using [Vayuvahana Technologies VayuAI](https://github.com/NamanMakkar/VayuAI) you can now calculate the speed of objects using [object tracking](../modes/track.md) alongside distance and time data, crucial for tasks like traffic monitoring and surveillance. The accuracy of speed estimation directly influences the efficiency and reliability of various applications, making it a key component in the advancement of intelligent systems and real-time decision-making processes.

## Advantages of Speed Estimation

- **Efficient Traffic Control:** Accurate speed estimation aids in managing traffic flow, enhancing safety, and reducing congestion on roadways.
- **Precise Autonomous Navigation:** In autonomous systems like self-driving cars, reliable speed estimation ensures safe and accurate vehicles navigation.
- **Enhanced Surveillance Security:** Speed estimation in surveillance analytics helps identify unusual behaviours or potential threats, improving the effectiveness of security measures.

## Usage Examples

!!! example "Speed Estimation using Vayuvahana Technologies VayuAI"

    === "CLI"

        ```bash
        # Run a speed example
        vajra solutions speed show=True

        # Pass a source video
        vajra solutions speed source="path/to/video.mp4"

        # Adjust meter per pixel value based on camera configuration
        vajra solutions speed meter_per_pixel=0.05
        ```

    === "Python"

        ```python
        import cv2

        from vajra import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("speed_management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Initialize speed estimation object
        speedestimator = solutions.SpeedEstimator(
            show=True,  # display the output
            model="vajra-v1-nano-det.pt",  # model weights
            fps=fps,  # adjust speed based on frame per second
            # max_speed=120,  # cap speed to a max value (km/h) to avoid outliers
            # max_hist=5,  # minimum frames object tracked before computing speed
            # meter_per_pixel=0.05,  # highly depends on the camera configuration
            # classes=[0, 2],  # estimate speed of specific classes.
            # line_width=2,  # adjust the line width for bounding boxes
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = speedestimator(im0)

            # print(results)  # access the output

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

### `SpeedEstimator` Arguments

Here's a table with the `SpeedEstimator` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "fps", "max_hist", "meter_per_pixel", "max_speed"]) }}

The `SpeedEstimator` solution allows the use of `track` parameters:

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Additionally, the following visualization options are supported:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## FAQ

### How do I estimate object speed using Vayuvahana Technologies VayuAI's VajraV1?

Estimating object speed with Vayuvahana Technologies VajraV1 involves combining object detection and tracking techniques. First, you need to detect objects in each frame using the VajraV1 model. Then, track these objects across frames to calculate their movement over time. Finally, use the distance travelled by the object between frames and the frame rate to estimate its speed.

**Example:**

```python
import cv2
from vajra import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

speedestimator = solutions.SpeedEstimator(
    model = "vajra-v1-nano-det.pt",
    show=True,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    results = speedestimator(im0)
    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```

### What are the benefits of using Vayuvahana Technologies VayuAI's VajraV1 for speed estimation in traffic management?

Using Vayuvahana Technologies VayuAI's VajraV1 for speed estimation offers significant advantages in traffic management:

- **Enhanced Safety:** Accurately estimate the vehicles speeds to detect over speeding and improve road safety.
- **Real-Time Monitoring:** Benefit from VajraV1's real-time object detection capability to monitor traffic flow and congestion effectively.
- **Scalability:** Deplot the model on various hardware setups, from edge devices to servers, ensuring flexible and scalable solutions for large-scale implementations.

### How accurate is speed estimation using VajraV1?

The accuracy of speed estimation using VajraV1 depends on several factors, including the quality of the object tracking, the resolution and the frame rate of the video, and environmental variables. While the speed estimator provides reliable estimates, it may not be 100% accurate due to variances in frame processing speed and object occlusion.

**Note:** Always consider margin of error and validate the estimates with ground truth data when possible.

For further accuracy improvement tips, check the [`SpeedEstimator` Arguments Section](#speedestimator-arguments).