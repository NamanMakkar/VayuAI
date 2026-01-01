# Object Counting using Vayuvahana Technologies VayuAI

## What is Object Counting?

Object counting with [Vayuvahana Technologies VayuAI](https://github.com/NamanMakkar/VayuAI/) involves accurate identification and counting of specific objects in vieos and camera streams. VajraV1 excels in real-time applications, providing efficient and precise object counting for various scenarious like crowd analysis and surveillance, thanks to its state of the art deep learning algorithms.

## Advantages of Object Counting

- **Enhanced Security and Surveillance:** Object counting enhances security and surveillance by accurately tracking and counting entities, aiding in ISR (Intelligence Surveillance and Reconnaissance) in defence and aiding in proactive threat detection in civilian environments.
- **Resource Optimization:** Object counting facilitates efficient resource management by providing accurate counts, optimizing resource allocation in applications like inventory management.
- **Informed Decision-Making:** Object counting offers valuable insights for decision-making, optimizing processes in retail, traffic management, and various other domains.

## Real World Applications

!!! example "Object Counting using Vayuvahana Technologies VajraV1"

    === "CLI"

        ```bash
        # Run a counting example
        vajra solutions count show=True

        # Pass a source video
        vajra solutions count source="path/to/video.mp4"

        # Pass region coordinates
        vajra solutions count region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]"
        ```

    === "Python"

        ```python
        import cv2

        from vajra import Vajra

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # region_points = [(20, 400), (1080, 400)]                                      # line counting
        region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # rectangle region
        # region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]   # polygon region

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Initialize object counter object
        counter = solutions.ObjectCounter(
            show=True,  # display the output
            region=region_points,  # pass region points
            model="vajra-v1-nano-det.pt",
            # classes=[0, 2],  # count specific classes i.e. person and car with COCO pretrained model.
            # tracker="botsort.yaml",  # choose trackers i.e "bytetrack.yaml"
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = counter(im0)

            # print(results)  # access the output

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

### `ObjectCounter` Arguments

Here's a table with the `ObjectCounter` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "show_in", "show_out", "region"]) }}

The `ObjectCounter` solution allows the use of several `track` arguments:

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Additionally, the visualization arguments listed below are supported:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## FAQ

### How do I count objects in a video using Vayuvahana Technologies VayuAI?

To count objects in a video using Vayuvahana Technologies VayuAI, follow these steps:

1. Import the necessary libraries (`cv2`, `vayuai`).
2. Define the counting region (e.g., a polygon, line, etc.).
3. Set up the video capture and initialize the object counter.
4. Process each frame to track objects and count them within the defined region.

Here's a simple example for counting in a region:

```python
import cv2

from vajra import solutions


def count_objects_in_region(video_path, output_video_path, model_path):
    """Count objects in a specific region within a video."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]
    counter = solutions.ObjectCounter(show=True, region=region_points, model=model_path)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or processing is complete.")
            break
        results = counter(im0)
        video_writer.write(results.plot_im)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


count_objects_in_region("path/to/video.mp4", "output_video.avi", "vajra-v1-nano-det.pt")
```

For more advanced configurations and options, check out the RegionCounter solution for counting objects in multiple regions simultaneously.

### What are the advantages of using Vayuvahana Technologies VayuAI for object counting?

1. **Enhanced Security and Surveillance:** Object counting enhances security and surveillance by accurately tracking and counting entities, aiding in ISR (Intelligence Surveillance and Reconnaissance) in defence and aiding in proactive threat detection in civilian environments.
2. **Resource Optimization:** Object counting facilitates efficient resource management by providing accurate counts, optimizing resource allocation in applications like inventory management.
3. **Informed Decision-Making:** Object counting offers valuable insights for decision-making, optimizing processes in retail, traffic management, and various other domains.
4. **Real-time Processing:** VajraV1's architecture enables real-time inference, making it suitable for live video streams and time-sensitive applications.

For implementation examples and practical applications, explore the TrackZone solution for tracking objects in specific zones.

### How can I count specific classes of objects using Vayuvahana Technologies VayuAI?

To count specific classes of objects using Vayuvahana Technologies VayuAI, you need to specify the classes you are interested in during the tracking phase. Below is a python example:

```python
import cv2

from vajra import solutions


def count_specific_classes(video_path, output_video_path, model_path, classes_to_count):
    """Count specific classes of objects in a video."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    line_points = [(20, 400), (1080, 400)]
    counter = solutions.ObjectCounter(show=True, region=line_points, model=model_path, classes=classes_to_count)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or processing is complete.")
            break
        results = counter(im0)
        video_writer.write(results.plot_im)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


count_specific_classes("path/to/video.mp4", "output_specific_classes.avi", "vajra-v1-nano-det.pt", [0, 2]) # person and car chosen as the classes
```

In this example, `classes_to_count=[0, 2]` means it counts objects of class `0` and `2` (e.g., person and car in the COCO dataset).

### Why should I use VajraV1 over other object detection models for real-time applications?

VajraV1 provides several advantages over other object detection models like YOLO12, YOLO11 or RT-DETR:

1. **Speed and Efficiency:** VajraV1 offers real-time processing capabilities, making it ideal for applications requiring high-speed inference, such as surveillance and autonomous driving.
2. **Accuracy:** VajraV1 provides state-of-the-art accuracy for object detection and tracking tasks reducing the number of false positives and improving overall system reliability.
3. **Ease of Integration:** VajraV1 offers seamless integration with various platforms and devices, including mobile and edge devices which is crucial for modern AI applications.
4. **Flexibility:** Supports various tasks like object detection, instance segmentation, and tracking with configurable models to meet specific use-case requirements.

### Can I use VajraV1 for advanced applications like crowd analysis, traffic management and ISR?

Yes, Vayavahana Technologies VayuAI is perfectly suited for applications like crowd analysis and traffic management due to its real-time detection capabilities, scalability and flexibility. Its advanced features allow for high-accuracy object tracking, counting and classification in dynamic environments. Example use cases:

- **Crowd Analysis:** Monitor and manage large gatherings, ensuring safety and optimizing crowd flow with region-based counting.
- **Traffic Management:** Track and count vehicles, analyze traffic patterns, and manage congestion in real-time with speed-estimation.
- **Intelligence Surveillance Reconnaissance:** Track and count enemy assets in real-time with the help of object counting.
- **Industrial Analytics:** Count products on conveyor belts and monitor production lines for quality control and efficiency improvements.
- **Retail Analytics:** Analyze customer movement patterns and product interactions to optimize store layouts and improve customer experience.