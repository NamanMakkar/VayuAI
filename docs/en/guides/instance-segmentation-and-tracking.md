# Instance Segmentation and Tracking using Vayuvahana Technologies VayuAI

## What is Instance Segmentation?

Instance segmentation is a computer vision task that involves identifying and outlining individual objects in an image at the pixel level. Unlike semantic segmentation which only classifies pixels by category, instance segmentation uniquely labels and precisely delineates each object instance, making it crucial for applications requiring detailed spatial understanding like medical imaging, autonomous driving and industrial automation.

[Vayuvahana Technologies VayuAI](https://github.com/NamanMakkar/VayuAI)'s VajraV1 provides powerful instance segmentation capabilities that enable precise object boundary detection while maintaining the speed and efficiency of the YOLO model family.

There are two types of instance segmentation tracking available in the VayuAI package:

- **Instance Segmentation with Class Objects:** Each class object is assigned a unique color for clear visual separation.

- **Instance Segmentation with Object Tracks:** Every track is represented by a distinct color, facilitating easy identification and tracking across video frames.

## Usage Example

!!! example "Instance segmentation using Vayuvahana Technologies VayuAI"

    === "CLI"

        ```bash
        # Instance segmentation using Vayuvahana Technologies VayuAI's VajraV1
        vajra solutions isegment show=True

        # Pass a source video
        vajra solutions isegment source="path/to/video.mp4"

        # Monitor the specific classes
        vajra solutions isegment classes="[0, 5]"
        ```

    === "Python"

        ```python
        import cv2

        from vajra import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("isegment_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Initialize instance segmentation object
        isegment = solutions.InstanceSegmentation(
            show=True,  # display the output
            model="vajra-v1-nano-seg.pt",  # model="vajra-v1-nano-seg.pt" for object segmentation using VajraV1.
            # classes=[0, 2],  # segment specific classes i.e, person and car with pretrained model.
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            results = isegment(im0)

            # print(results)  # access the output

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

## InstanceSegmentation Arguments

Here's a table with the `InstanceSegmentation` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "region"]) }}

You can also take advantage of `track` arguments within the `InstanceSegmentation` solution:

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Moreover, the following visualization arguments are available:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## Applications of Instance Segmentation and Tracking

Instance segmentation with VajraV1 has numerous real-world applications across various industries:

### Autonomous Vehicles

In self-driving cars, instance segmentation helps identify and track pedestrians, vehicles, traffic signs, and other road elements at the pixel level. This precise understanding of the environment is crucial for navigation and safety decisions. VajraV1's real-time performance makes it ideal for time-sensitive applications.

### Medical Imaging

Instance segmentation can identify and outline tumours, organs or cellular structures in medical scans. VajraV1's ability to precisely delineate object boundaries makes it valuable for medical diagnostics and treatment planning.

### Construction Site Monitoring

At construction sites, instance segmentatio can track heavy machinery, workers and materials. This helps ensure safety by monitoring equipment positions and detecting when workers enter hazardous areas, while also optimizing workflow and resource allocation.

### Waste Management and Recycling

VajraV1 can be used in waste management facilities to identify and sort different types of materials. The model can segment plastic waste, cardboard, metal and other recyclables with high precision, enabling automated sorting systems to process waste more efficiently.

## FAQ

### How do I perform instance segmentation using Vayuvahana Technologies VayuAI?

To perform instance segmentation using Vayuvahana Technologies VayuAI's VajraV1, initialize the VajraV1 model with a segmentation version of VajraV1 and process video frames through it. Here is a simplified code example:

```python
import cv2

from vajra import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("instance-segmentation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init InstanceSegmentation
isegment = solutions.InstanceSegmentation(
    show=True,  # display the output
    model="vajra-v1-nano-seg.pt",  # model="vajra-v1-nano-seg.pt" for object segmentation using VajraV1.
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break
    results = isegment(im0)
    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```

### What is the difference between instance segmentation and object tracking?

Instance segmentation identifies and outlines individual objects within an image, giving each object a unique label and mask. Object tracking extends this by assigning consistent IDs to objects across video frames, facilitating continuous tracking of the same objects over time. When combined, you get powerful capabilities for analysing object movement and behaviour in videos while maintaining precise boundary information.

