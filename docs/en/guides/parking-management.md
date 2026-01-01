# Parking Management Using Vayuvahana Technologies VayuAI's VajraV1

## Parking Management System

Parking management with VajraV1 ensures efficient and safe parking by organizing spaces and monitoring availability. VajraV1 can optimize parking lot management through real-time vehicle detection and insights into parking occupancy.

## Advantages of Parking Management System

- **Efficiency**: Parking lot management optimizes the use of parking spaces and reduces congestion.
- **Safety and Security**: Parking management using YOLO11 improves the safety of both people and vehicles through surveillance and security measures.
- **Reduced Emissions**: Parking management using YOLO11 manages traffic flow to minimize idle time and emissions in parking lots.

## Parking Management System Code Workflow

**Step-1:** Capture a frame from the video or camera stream where you want to manage the parking lot.

**Step-2:** Use the provided code to launch a graphical interface, where you can select an image and start outlining parking regions by mouse click to create polygons.


!!! example "Parking Annotator"

    ??? note "Additional step for installing `tkinter`"

        Generally, `tkinter` comes pre-packaged with Python. However, if it did not, you can install it using the highlighted steps:

        - **Linux**: (Debian/Ubuntu): `sudo apt install python3-tk`
        - **Fedora**: `sudo dnf install python3-tkinter`
        - **Arch**: `sudo pacman -S tk`
        - **Windows**: Reinstall Python and enable the checkbox `tcl/tk and IDLE` on **Optional Features** during installation
        - **MacOS**: Reinstall Python from [https://www.python.org/downloads/macos/](https://www.python.org/downloads/macos/) or `brew install python-tk`

    === "Python"

        ```python
        from vajra import solutions

        solutions.ParkingPtsSelection()
        ```

**Step-3:** After defining the parking areas with polygons, click `save` to store a JSON file with the data in your working directory.

**Step-4:** You can now utilize the provided code for parking management.

!!! example "Parking Management"

    === "Python"

        ```python
        import cv2
        from vajra import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("parking-management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        parkingmanager = solutions.ParkingManagement(
            model="vajra-v1-nano-det.pt",
            json_file="bounding_boxes.json" # path to parking annotations file
        )

        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = parkingmanager(im0)

            print(results)

            video_writer.write(results.plot_im)
        
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

## `ParkingManagement` Arguments

Here's a table with the `ParkingManagement` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "json_file"]) }}

The `ParkingManagement` solution allows the use of several `track` parameters:

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Moreover, the following visualization options are supported:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width"]) }}

## FAQ

### How can I define parking spaces using VajraV1?

Defining parking spaces is straightforward with VajraV1:

1. Capture a frame from a video or camera stream.
2. Use the provided code to launch a GUI for selecting an image and drawing polygons to define parking spaces.
3. Save the labeled data in JSON format for further processing. For comprehensive instructions, check the selection of points section above.

### What are some real-world applications of Vayuvahana Technologies VayuAI's VajraV1 in parking lot management?

VajraV1 can be utilized in various real-world applications for parking lot management, including:

- **Parking Space Detection**: Accurately identifying available and occupied spaces.
- **Surveillance**: Enhancing security through real-time monitoring.
- **Traffic Flow Management**: Reducing idle times and congestion with efficient traffic handling. Images showcasing these applications can be found in [real-world applications](#real-world-applications).