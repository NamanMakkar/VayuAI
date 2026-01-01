# Security Alarm System Using Vayuvahana Technologies VayuAI

The Security Alarm System Solution utilizing Vayuvahana Technologies VayuAI's VajraV1 integrates advanced computer vision capabilities to enhance security measures. VajraV1, developed by Vayuvahana Technologies, provides real-time object detection, allowing the system to identify and respond to potential security threats promptly. This solution offers several advantages:

- **Real-time Detection:** VajraV1's efficiency enables the Security Alarm System to detect and respond to security incidents in real-time, minimizing response time.
- **Accuracy:** VajraV1 is known for its accuracy in object detection, reducing false positives and enhancing the reliability of the security alarm system.
- **Integration Capabilities:** The project can be seamlessly integrated with existing security infrastructure, providing an upgraded layer of intelligent surveillance.

???+ note

    App Password Generation is necessary

- For Gmail, navigate to [Google's App Password Generator](https://myaccount.google.com/apppasswords), designate an app name such as "security project," and obtain a 16-digit password. Copy this password and paste it into the designated `password` field in the code below.

!!! example "Security Alarm System using Vayuvahana Technologies VayuAI"

    === "Python"

        ```python
        import cv2

        from vajra import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("security_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        
        from_email = "abc@gmail.com" # Sender's email address
        password = "---- ---- ---- ----" # 16-digits password generated via the app password generator
        to_email = "xyz@gmail.com"

        security_alarm = solutions.SecurityAlarm(
            show=True, # display the output
            model="vajra-v1-nano-det.pt", # pretrained model
            records=1, # total detections count required to send an email
        )

        security_alarm.authenticate(from_email, password, to_email)

        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            results = security_alarm(im0)
            video_writer.write(results.plot_im)
        
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

When you execute this code, you will receive a single notification on your email if any object is detected. The notification is sent immediately, not repeatedly. However, feel free to customize the code to suit your project requirements.

### `SecurityAlarm` Arguments

Here's a table with the `SecurityAlarm` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "records"]) }}

The `SecurityAlarm` solution supports a variety of `track` parameters:

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Moreover, the following visualization settings are available:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## How It Works

The Security Alarm System uses object tracking to monitor video feeds and detect potential security threats. When the system detects objects that exceed the specified threshold (set by the `records` parameter), it automatically sends an email notification with an image attachment showing the detected objects.

The system leverages the SecurityAlarm class which provides methods to:

1.Process frames and extract object detections
2. Annotate frames with bounding boxes around detected objects
3. Send email notifications when detection thresholds are exceeded

This implementation is ideal for home security, retail surveillance, and other monitoring applications where immediate notification of detected objects is critical.

## FAQ

### How does Vayuvahana Technologies VayuAI improve the accuracy of a security alarm system?

Vayuvahana Technologies VayuAI's VajraV1 enhances security alarm systems by delivering high-accuracy, real-time object detection. Its advanced algorithms reduce false positives, ensuring that the system only responds to genuine threats. This increased reliability can be seamlessly integrated with existing security infrastructure, upgrading the overall surveillance quality.

### Can I integrate Vayuvahana Technologies VayuAI with my existing security infrastructure?

Yes, Vayuvahana Technologies VayuAI can be seamlessly integrated with your existing security infrastructure. The system supports various modes and provides flexibility for customization, allowing you to enhance your existing setup with advanced object detection capabilities.

### What are the storage requirements for running VajraV1?

Running Vayuvahana Technologies VayuAI's VajraV1 on a standard setup requires around 5GB of disk space. This includes space for storing the VajraV1 model and any additional dependencies.

### What makes Vayuvahana Technologies VayuAI's VajraV1 different from other object detection models like Faster R-CNN, SSD or YOLO models?

Vayuvahana Technologies VajraV1 provides an edge over models like Faster R-CNN or SSD with its real-time detection capabilities and higher accuracy. While being more accurate than the YOLO11 and YOLO12 with competitive latency. This makes the VajraV1 ideal for time-sensitive applications like security alarm systems.

### How can I reduce the frequency of false positives in my security system using VajraV1?

To reduce the false positives, ensure your Vayuvahana Technologies VajraV1 model is adequately trained with a diverse and well-annotated dataset. Fine-tuning hyperparameters and regularly updating the model with new data can significantly improve detection accuracy.
