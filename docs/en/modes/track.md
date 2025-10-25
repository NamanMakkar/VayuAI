# Multi-Object Tracking with Vayuvahana Technlogies VayuAI

Object tracking in the realm of video analytics is a critical task that not only identifies the location and class of objects within the frame but also maintains a unique ID for each detected object as the video progresses. The applications are limitless—ranging from surveillance and security to real-time sports analytics.

## Why Choose Vayuvahana Technologies VayuAI for Object Tracking?

The output from the trackers used in the VayuAI SDK provides object IDs. This makes it easy to track objects in video streams and perfrom subsequent analytics. Here's why you should consider using Vayuvahana Technologies VayuAI for your object tracking needs:

- **Efficiency:** Process video streams in real-time without compromising accuracy.
- **Flexibility:** Supports multiple tracking algorithms and configurations.
- **Ease of Use:** Simple Python API and CLI options for quick integration and deployment.
- **Customizability:** Easy to use with custom trained Vajra models, allowing integration into domain-specific applications.

## Features

Vayuvahana Technologies VayuAI extends its object detection features to provide robust and versatile object tracking:

- **Real-Time Tracking:** Seamlessly track objects in high-frame-rate videos.
- **Multiple Tracker Support:** Choose from a variety of established tracking algorithms.
- **Customizable Tracker Configurations:** Tailor the tracking algorithm to meet specific requirements by adjusting various parameters.

## Available Trackers

Vayuvahana Technologies VayuAI supports the following tracking algorithms. They can be enabled by passing the relevant YAML configuration file such as `tracker=tracker_type.yaml`:

- [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - Use `botsort.yaml` to enable this tracker.
- [ByteTrack](https://github.com/FoundationVision/ByteTrack) - Use `bytetrack.yaml` to enable this tracker.

The default tracker is BoT-SORT.

## Tracking

To run the tracker on video streams, use a trained Detect, Segment or Pose model such as VajraV1-nano-det, VajraV1-nano-seg and VajraV1-nano-pose.

!!! example

    === "Python"

        ```python
        from vajra import Vajra

        # Load an official or custom model
        model = Vajra("vajra-v1-nano-det.pt")  # Load an official Detect model
        model = Vajra("vajra-v1-nano-seg.pt")  # Load an official Segment model
        model = Vajra("vajra-v1-nano-pose.pt")  # Load an official Pose model
        model = Vajra("path/to/best-vajra-v1-nano-det.pt")  # Load a custom trained model

        # Perform tracking with the model
        results = model.track("https://youtu.be/LNwODJXcvt4", show=True)  # Tracking with default tracker
        results = model.track("https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # with ByteTrack
        ```

    === "CLI"

        ```bash
        # Perform tracking with various models using the command line interface
        vajra track model=vajra-v1-nano-det.pt source="https://youtu.be/LNwODJXcvt4"      # Official Detect model
        vajra track model=vajra-v1-nano-seg.pt source="https://youtu.be/LNwODJXcvt4"  # Official Segment model
        vajra track model=vajra-v1-nano-pose.pt source="https://youtu.be/LNwODJXcvt4" # Official Pose model
        vajra track model=path/to/best-vajra-v1-nano-det.pt source="https://youtu.be/LNwODJXcvt4" # Custom trained model

        # Track using ByteTrack tracker
        vajra track model=path/to/best-vajra-v1-nano-det.pt tracker="bytetrack.yaml"
        ```

As can be seen in the above usage, tracking is available for all Detect, Segment and Pose models run on videos or streaming sources.

## Configuration

### Tracking Arguments

Tracking configuration shares properties with Predict mode, such as `conf`, `iou`, and `show`. For further configurations, refer to the [Predict](../modes/predict.md#inference-arguments) model page.

!!! example

    === "Python"

        ```python
        from vajra import Vajra

        # Configure the tracking parameters and run the tracker
        model = Vajra("vajra-v1-nano-det.pt")
        results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.3, iou=0.5, show=True)
        ```

    === "CLI"

        ```bash
        # Configure tracking parameters and run the tracker using the command line interface
        vajra track model=vajra-v1-nano-det.pt source="https://youtu.be/LNwODJXcvt4" conf=0.3, iou=0.5 show
        ```

### Tracker Selection

In order to modify the tracker configuration file simply make a copy of the tracker config file from [vajra/configs/trackers](https://github.com/NamanMakkar/VayuAI/configs/trackers) and modify any configurations (except the `tracker_type`) as per your needs.

!!! example

    === "Python"

        ```python
        from vajra import Vajra

        # Load the model and run the tracker with a custom configuration file
        model = Vajra("vajra-v1-nano-det.pt")
        results = model.track(source="https://youtu.be/LNwODJXcvt4", tracker="custom_tracker.yaml")
        ```

    === "CLI"

        ```bash
        # Load the model and run the tracker with a custom configuration file using the command line interface
        vajra track model=vajra-v1-nano-det.pt source="https://youtu.be/LNwODJXcvt4" tracker='custom_tracker.yaml'
        ```

Refer to [Tracker Arguments](#tracker-arguments) section for a detailed description of each parameter.

### Tracker Arguments

Tracking behaviours can be fine-tuned by editing the YAML configuration files specific to each tracking algorithm. These files define parameters like thresholds, buffers and matching logic:

- [botsort.yaml](https://github.com/NamanMakkar/VayuAI/tree/main/vajra/configs/trackers/botsort.yaml)
- [bytetrack.yaml](https://github.com/NamanMakkar/VayuAI/tree/main/vajra/configs/trackers/bytetrack.yaml)

The following table provides a description of each parameter:

!!! warning "Tracker Threshold Information"

    If object confidence score will be low, i.e lower than [`track_high_thresh`](https://github.com/NamanMakkar/VayuAI/tree/main/vajra/configs/trackers/bytetrack.yaml#L2), then there will be no tracks successfully returned and updated.

| **Parameter**       | **Valid Values or Ranges**                    | **Description**                                                                                                                                        |
| ------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `tracker_type`      | `botsort`, `bytetrack`                        | Specifies the tracker type. Options are `botsort` or `bytetrack`.                                                                                      |
| `track_high_thresh` | `0.0-1.0`                                     | Threshold for the first association during tracking used. Affects how confidently a detection is matched to an existing track.                         |
| `track_low_thresh`  | `0.0-1.0`                                     | Threshold for the second association during tracking. Used when the first association fails, with more lenient criteria.                               |
| `new_track_thresh`  | `0.0-1.0`                                     | Threshold to initialize a new track if the detection does not match any existing tracks. Controls when a new object is considered to appear.           |
| `track_buffer`      | `>=0`                                         | Buffer used to indicate the number of frames lost tracks should be kept alive before getting removed. Higher value means more tolerance for occlusion. |
| `match_thresh`      | `0.0-1.0`                                     | Threshold for matching tracks. Higher values makes the matching more lenient.                                                                          |
| `fuse_score`        | `True`, `False`                               | Determines whether to fuse confidence scores with IoU distances before matching. Helps balance spatial and confidence information when associating.    |
| `gmc_method`        | `orb`, `sift`, `ecc`, `sparseOptFlow`, `None` | Method used for global motion compensation. Helps account for camera movement to improve tracking.                                                     |
| `proximity_thresh`  | `0.0-1.0`                                     | Minimum IoU required for a valid match with ReID (Re-identification). Ensures spatial closeness before using appearance cues.                          |
| `appearance_thresh` | `0.0-1.0`                                     | Minimum appearance similarity required for ReID. Sets how visually similar two detections must be to be linked.                                        |
| `with_reid`         | `True`, `False`                               | Indicates whether to use ReID. Enables appearance-based matching for better tracking across occlusions. Only supported by BoTSORT.

### Enabling Re-Identification (ReID)

By default, ReID is turned off to minimize performance overhead. Enabling it is simple, just set `with_reid: True` in the [tracker configuration](https://github.com/NamanMakkar/VayuAI/tree/main/vajra/configs/trackers/botsort.yaml). ReID leverages features directly from the VajraV1 detector, adding minimal overhead. It is ideal when you need some level of ReID without significantly impacting performance.

## Python Examples

### Persisting Tracks Loop

Here is a Python script using OpenCV (`cv2`) and VajraV1 to run object tracking on video frames. This script assumes you have already installed the necessary packages (`opencv-python` and `vayuai`). The `persist=True` argument tells the tracker that the current image or frame is the next in a sequence and to expect tracks form the previous image in the current image.

!!! example "Streaming for-loop with tracking"

    ```python
    import cv2

    from vajra import Vajra

    # Load the VajraV1 model
    model = Vajra("vajra-v1-nano-det.pt")

    # Open the video file
    video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run VajraV1 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("VajraV1 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    ```

Please note the change from model(frame)` to `model.track(frame)`, which enables object tracking instead of simple detection. This modified script will run the tracker on each frame of the video, visualize the results, and display them in a window. The loop can be exited by pressing 'q'.

### Plotting Tracks Over Time

Visualizing object tracks over consecutive frames can provide valuable insights into the movement patterns and behaviour of detected objects within a video. With Vayuvahana Technologies VayuAI, plotting these tracks is a seamless and efficient process.

In the following example, we demonstrate how to utilize VayuAI's tracking capabilities to plot the movement of detected objects across multiple video frames. This script involves opening a video file, reading it frame by frame, and utilizing the VajraV1 model to identify and track various objects. By retaining the center points of the detected bounding boxes and connecting them we can draw lines that represent the paths followed by the tracked objects.

!!! example "Plotting tracks over multiple video frames"

    ```python
    from collections import defaultdict

    import cv2
    import numpy as np

    from vajra import Vajra

    # Load the VajraV1 model
    model = Vajra("vajra-v1-nano-det.pt")

    # Open the video file
    video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run VajraV1 tracking on the frame, persisting tracks between frames
            result = model.track(frame, persist=True)[0]

            # Get the boxes and track IDs
            if result.boxes and result.boxes.is_track:
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()

                # Visualize the result on the frame
                frame = result.plot()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Display the annotated frame
            cv2.imshow("VajraV1 Tracking", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    ```

### Multithreaded Tracking

Multithreaded tracking provides the capability to run object tracking on multiple video streams simultaneously. This is particularly useful when handling multiple video inputs, such as from multiple surveillance cameras, where concurrent processing can greatly enhance efficiency and performance.

In the provided Python script, we make use of Python's `threading` module to run multiple instances of the tracker concurrently. Each thread is responsible for running the tracker on one video file, and all the threads run simultaneously in the background.

To ensure that each thread receives the correct parameters (the video file, the model to use and the file index), we define a function `run_tracker_in_thread` that accepts these parameters and contains the main tracking loop. This function reads the video frame by frame, runs the tracker, and displays the results.

Two different models are used in this example: `vajra-v1-nano-det.pt` and `vajra-v1-nano-seg.pt`, each tracking objects in a different video file. The video files are specified in `SOURCES`.

The `daemon=True` parameter in `threading.Thread` means that these threads will be closed as soon as the main program finishes. We then start the threads with `start()` and use `join()` to make the main thread wait until both tracker threads have finished.

Finally, after all threads have completed their task, the windows displaying the results are closed using `cv2.destroyAllWindows()`.

!!! example "Multithreaded tracking implementation"

    ```python
    import threading

    import cv2

    from vajra import Vajra

    # Define model names and video sources
    MODEL_NAMES = ["vajra-v1-nano-det.pt", "vajra-v1-nano-seg.pt"]
    SOURCES = ["path/to/video.mp4", "0"]  # local video, 0 for webcam


    def run_tracker_in_thread(model_name, filename):
        """
        Run Vajra tracker in its own thread for concurrent processing.

        Args:
            model_name (str): The Vajra model object.
            filename (str): The path to the video file or the identifier for the webcam/external camera source.
        """
        model = Vajra(model_name)
        results = model.track(filename, save=True, stream=True)
        for r in results:
            pass


    # Create and start tracker threads using a for loop
    tracker_threads = []
    for video_file, model_name in zip(SOURCES, MODEL_NAMES):
        thread = threading.Thread(target=run_tracker_in_thread, args=(model_name, video_file), daemon=True)
        tracker_threads.append(thread)
        thread.start()

    # Wait for all tracker threads to finish
    for thread in tracker_threads:
        thread.join()

    # Clean up and close windows
    cv2.destroyAllWindows()
    ```

This example can easily be extended to handle more video files and models by creating more threads and applying the same methodology.

## FAQ

### What is Multi-Object Tracking and how does Vayuvahana Technologies VayuAI support it?

Multi-object tracking in video analytics involves both identifying objects and maintaining a unique ID for each detected object across video frames. Vayuvahana Technologies VayuAI supports this by providing real-time tracking along with object IDs, facilitating tasks such as security surveillance and sports analytics. The system uses trackers like [BoT-SORT](https://github.com/NirAharon/BoT-SORT) and [ByteTrack](https://github.com/FoundationVision/ByteTrack), which can be configured via YAML files.

### How do I configure a custom tracker for Vayuvahana Technologies VayuAI?

You can configure a custom tracker by copying an existing tracker configuration file (e.g., `custom_tracker.yaml`) from the [VayuAI tracker configuration directory](https://github.com/NamanMakkar/VayuAI/tree/main/vajra/configs/trackers) and modifying parameters as needed, except for the `tracker_type`. Use this file in your tracking model like so:

!!! example

    === "Python"

        ```python
        from vajra import Vajra

        model = Vajra("vajra-v1-nano-det.pt")
        results = model.track(source="https://youtu.be/LNwODJXcvt4", tracker="custom_tracker.yaml")
        ```

    === "CLI"

        ```bash
        vajra track model=vajra-v1-nano-det.pt source="https://youtu.be/LNwODJXcvt4" tracker='custom_tracker.yaml'
        ```

### How can I run object tracking on multiple video streams simultaneously?

To run object tracking on multiple video streams simultaneously, you can use Python's `threading` module. Each thread will handle a separate video stream. Here's an example of how you can set this up:

!!! example "Multithreaded tracking implementation"

    ```python
    import threading

    import cv2

    from vajra import Vajra

    # Define model names and video sources
    MODEL_NAMES = ["vajra-v1-nano-det.pt", "vajra-v1-nano-seg.pt"]
    SOURCES = ["path/to/video.mp4", "0"]  # local video, 0 for webcam


    def run_tracker_in_thread(model_name, filename):
        """
        Run Vajra tracker in its own thread for concurrent processing.

        Args:
            model_name (str): The Vajra model object.
            filename (str): The path to the video file or the identifier for the webcam/external camera source.
        """
        model = Vajra(model_name)
        results = model.track(filename, save=True, stream=True)
        for r in results:
            pass


    # Create and start tracker threads using a for loop
    tracker_threads = []
    for video_file, model_name in zip(SOURCES, MODEL_NAMES):
        thread = threading.Thread(target=run_tracker_in_thread, args=(model_name, video_file), daemon=True)
        tracker_threads.append(thread)
        thread.start()

    # Wait for all tracker threads to finish
    for thread in tracker_threads:
        thread.join()

    # Clean up and close windows
    cv2.destroyAllWindows()
    ```

### What are the real-world applications of multi-object tracking with Vayuvahana Technologies VayuAI?

Multi-object tracking with Vayuvahana Technologies VayuAI has numerous applications, including:

- **Defence & Aerospace:** Target tracking from UAV's or satellites for reconnaissance.
- **Security Systems:** Monitoring suspicious activities and creating security alarms.
- **Transportation:** Vehicle tracking for traffic management and autonomous driving.
- **Retail:** People tracking for in-store analytics and security.
- **Aquaculture:** Fish tracking for monitoring aquatic environments.
- **Sports Analytics:** Tracking players and equipment for performance analysis.

These applications benefit from Vayuvahana Technologies VayuAI's ability to process high-frame-rate videos in real time with exceptional accuracy.

### How can I visualize object tracks over multiple video frames with Vayuvahana Technologies VayuAI?

To visualize object tracks over multiple video frames, you can use the object tracking features along with OpenCV to draw the paths of the detected objects. Here's an example script that demonstrates this:

!!! example "Plotting tracks over multiple video frames"

    ```python
    from collections import defaultdict

    import cv2
    import numpy as np

    from vajra import Vajra

    model = Vajra("vajra-v1-nano-det.pt")

    video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)

    track_history = defaultdict(lambda: [])

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            result = model.track(frame, persist=True)[0]

            if result.boxes and result.boxes.is_track:
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()

                frame = result.plot()

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 30:
                        track.pop(0)

                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            cv2.imshow("VajraV1 Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    ```




