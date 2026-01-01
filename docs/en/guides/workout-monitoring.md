# Monitor your workouts with VajraV1

Monitoring workouts through pose estimation with the VajraV1 models enhances exercise assessment by accurately tracking key body landmarks and joints in real-time. This technology provides instant feedback on exercise form, tracks workout routines, and measures performance metrices, optimizing training sessions for users and trainers.

## Advantages of Workouts Monitoring

- **Optimized Performance:** Tailoring workouts based on monitoring data for better results.
- **Goal Achievement:** Track and adjust fitness goals for measurable progress.
- **Personalization:** Customized workout plans based on individual data for effectiveness.
- **Health Awareness:** Early detection of patterns indicating health issues or over-training.
- **Informed Decisions:** Data-driven decisions for adjusting routines and setting realistic goals.

## Usage Example

!!! example "Workout Monitoring using VajraV1"

    === "CLI"

        ```bash

        vajra solutions workout show=True

        vajra solutions workout source="path/to/video.mp4"

        vajra solutions workout kpts="[6, 8, 10]"
        ```

    === "Python"

        ```python
        import cv2
        from vajra import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("workouts_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        gym = solutions.AIGym(
            show=True,
            kpts=[6, 8, 10],  # keypoints for monitoring specific exercise, by default it's for pushup
            model="vajra-v1-nano-pose.pt",
        )

        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = gym(im0)

            # print(results)  # access the output

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

## AIGym Arguments

Here's a table with the `AIGym` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "up_angle", "down_angle", "kpts"]) }}

The `AIGym` solution also supports a range of object tracking parameters:

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Additionally, the following visualization settings can be applied:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## FAQ

### How do I monitor my workouts with VajraV1?

To monitor your workouts with VajraV1, you can utilize the pose estimation capabilities to track and analyze key body landmarks and joints in real-time. This allows you to receive instant feedback on your exercise form, count repetitions, and measure performance metrics. You can start by using the provided example code for push-ups, pull-ups, or ab workouts as shown:

```python
import cv2

from vajra import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

gym = solutions.AIGym(
    line_width=2,
    show=True,
    kpts=[6, 8, 10],
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break
    results = gym(im0)

cv2.destroyAllWindows()
```

For further customization you can refer to the [AIGym](#aigym-arguments) section in the documentation.

### What are the benefits of using VajraV1 for workout monitoring?

- **Optimized Performance:** By tailoring workouts based on monitoring data, you can achieve better results.
- **Goal Achievement:** Easily track and adjust fitness goals for measurable progress.
- **Personalization:** Get customized workout plans based on your individual data for optimal effectiveness.
- **Health Awareness:** Early detection of patterns that indicate potential health issues or over-training.
- **Informed Decisions:** Make data-driven decisions to adjust routines and set realistic goals.