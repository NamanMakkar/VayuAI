# VisionEye View Object Mapping with VajraV1

## What is VisionEye Object Mapping?

VajraV1 VisionEye offers the capability for computers to identify and pinpoint objects, simulating the observational precision of the human eye. This functionality enables computers to discern and focus on specific objects, much like the way the human eye observes details from a particular viewpoint.

!!! example "VisionEye Mapping using VajraV1"

    === "CLI"

        ```bash
        # Monitor objects position with visioneye
        vajra solutions visioneye show=True

        # Pass a source video
        vajra solutions visioneye source="path/to/video.mp4"

        # Monitor the specific classes
        vajra solutions visioneye classes="[0, 5]"
        ```

    === "Python"

        ```python
        import cv2
        from vajra import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("visioneye_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Initialize vision eye object
        visioneye = solutions.VisionEye(
            show=True,  # display the output
            model="vajra-v1-nano-det.pt",
            classes=[0, 2],
            vision_point=(50, 50),
        )

        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            results = visioneye(im0)

            print(results)  # access the output

            video_writer.write(results.plot_im)  # write the video file

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

## VisionEye Arguments

Here's a table with the `VisionEye` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "vision_point"]) }}

You can also utilize various `track` arguments within the `VisionEye` solution:

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Furthermore, some visualization arguments are supported, as listed below:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## How VisionEye Works

VisionEye works by establishing a fixed vision point in the frame and drawing lines from this point to detected objects. This simulates how human vision focuses on multiple objects from a single viewpoint. The solution uses object tracking to maintain consistent identification of objects across frames, creating a visual representation of the spatial relationship between the observer (vision point) and the objects in the scene.

The `process` method in the VisionEye class performs several key operations:

1. Extracts tracks (bounding boxes, classes, and masks) from the input image
2. Creates an annotator to draw bounding boxes and labels
3. For each detected object, draws a box label and creates a vision line from the vision point
4. Returns the annotated image with tracking statistics

This approach is particularly useful for applications requiring spatial awareness and object relationship visualization, such as surveillance systems, autonomous navigation, and interactive installations.


## Applications of VisionEye

VisionEye objects mapping has numerous practical applications across various industries:

- **Security and Surveillance**: Monitor multiple objects of interest from a fixed camera position
- **Retail Analytics**: Track customer movement patterns in relation to store displays
- **Sports Analysis**: Analyze player positioning and movement from a coach's perspective
- **Autonomous Vehicles**: Visualize how a vehicle "sees" and prioritizes objects in its environment
- **Human-Computer Interaction**: Create more intuitive interfaces that respond to spatial relationships

By combining VisionEye with other VayuAI solutions like distance calculations or speed estimation, you can build comprehensive systems that not only track objects but also understand their spatial relationships and behaviours.

