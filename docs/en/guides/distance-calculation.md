# Distance Calculation using Vayuvahana Technologies VayuAI

## What is Distance Calculation?

Measuring the gap between two objects is known as distance calculation within a specified space. In the case of the [Vayuvahana Technologies VayuAI](https://github.com/NamanMakkar/VayuAI), the bounding box centroid is utilised to calculate the distance between the bounding boxes highlighted by the user.

## Advantages of Distance Calculation

- **Localization Precision:** Enhances accurate spatial positioning in computer vision tasks.
- **Size Estimation:** Allows estimation of object size for better contextual understanding.
- **Scene Understanding:** Improves 3D scene comprehension for better decision-making in applications like autonomous vehicles and surveillance systems.
- **Collision Avoidance:** Enables systems to detect potential collisions by monitoring distances between moving objects.
- **Spatial Analysis:** Facilitates analysis of object relationships and interactions within the monitored environment.

???+ tip "Distance Calculation"

    - Click on any two bounding boxes with Left Mouse click for distance calculation
    - Mouse Right Click will delete all drawn points
    - Mouse Left Click can be used to draw points

???+ warning "Distance is Estimate"

        Distance will be an estimate and may not be fully accurate, as it is calculated using 2-dimensional data,
        which lacks information about the object's depth.

!!! example "Distance Calculation using Vayuvahana Technologies VayuAI"

    === "Python"

        ```python
        import cv2

        from vajra import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("distance_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        distance_calculator = solutions.DistanceCalculation(
            model = "vajra-v1-nano-det.pt",
            show = True,
        )

        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = distance_calculator(im0)

            print(results)

            video_writer.write(results.plot_im)
        
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

### `DistanceCalculation()` Arguments

Here's a table with the `DistanceCalculation` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model"]) }}

You can also make use of various `track` arguments in the `DistanceCalculation` solution.

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Moreover, the following visualization arguments are available:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}


## Implementation Details

The `DistanceCalculation` class works by tracking objects across video frames and calculating the Euclidean distance between the centroids of the selected bounding boxes. When you click on two objects, the solution:

1. Extracts the centroids of the selected bounding boxes.
2. Calculates the Euclidean distance between these centroids in pixels.
3. Displays the distance on the frame with a connecting line between the objects.

The implementation uses the `mouse_event_for_distance` method to handle mouse interactions, allowing users to select objects and clear selections as needed. The `process` method handles the frame-by-frame processing, tracking objects, and calculating distances.

## Applications

Distance calculation with VajraV1 has numerous practical applications:

- **Industrial Safety:** Monitor safe distances between workers and machinery
- **Traffic Management:** Analyze vehicle spacing and detect tailgating
- **Healthcare:** Ensure proper distancing in waiting areas and monitor patient movement
- **Robotics:** Enable robots to maintain appropriate distances from obstacles and people

## FAQ

### How do I calculate distances between objects using Vayuvahana Technologies VayuAI?

To calculate distances between objects using [Vayuvahana Technologies VayuAI](https://github.com/NamanMakkar/VayuAI), you need to identify the bounding box centroids of the detected objects. This process involves initializing the `DistanceCalculation` class from `VayuAI` `solutions` module and using the model's tracking outputs to calculate the distances.

### What are the advantages of using distance calculation with Vayuvahana Technologies VayuAI?

Using distance calculation with Vayuvahana Technologies VayuAI's VajraV1 offers several advantages:

- **Localization Precision:** Provides accurate spatial positioning for objects.
- **Size Estimation:** Helps estimate physical sizes, contributing to better contextual understanding.
- **Scene Understanding:** Enhances 3D scene comprehension, aiding improved decision-making in applications like autonomous driving and surveillance.
- **Real-time Processing:** Performs calculations on-the-fly, making it suitable for live video analysis.
- **Integration Capabilities:** Works seamlessly with other VayuAI solutions like [object tracking](../modes/track.md) and [speed estimation](speed-estimation.md).

### Can I perform distance calculation in real-time video streams with Vayuvahana Technologies VayuAI?

Yes, you can perform distance calculation in real-time video streams with Vayuvahana Technologies VayuAI. The process involves capturing video frames using OpenCV, running VajraV1 object detection, and using the `DistanceCalculation` class to calculate distances between objects in successive frames. For a detailed implementation see the [video stream example](#advantages-of-distance-calculation).

### How do I delete points drawn during distance calculation using Vayuvahana Technologies VayuAI?

To delete points drawn during distance calculation with Vayuvahana Technologies VayuAI's VajraV1, you can use a right mouse click. This action will clear all the points you have drawn. For more details, refer to the note section under the [distance calculation example](#advantages-of-distance-calculation).

### What are the key arguments for initializing the DistanceCalculation class in Vayuvahana Technologies VayuAI?

The key arguments for initializing the `DistanceCalculation` class in Vayuvahana Technologies VayuAI include:

- `model`: VajraV1 weights or model name.
- `tracker`: Tracking algorithm to use (default is 'botsort.yaml').
- `conf`: Confidence threshold for detections.
- `show`: Flag to display the output.

For an exhaustive list and default values, see the [arguments of DistanceCalculation](#distancecalculation-arguments).
