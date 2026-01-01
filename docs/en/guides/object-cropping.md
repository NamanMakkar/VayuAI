# Object Cropping using Vayuvahana Technologies VayuAI

## What is Object Cropping?

Object cropping with [Vayuvahana Technologies VayuAI's VajraV1](https://github.com/NamanMakkar/VayuAI) involves isolating and extracting specific detected objects from an image or video. The VajraV1 model capabilities are utilized to accurately identify and delineate objects, enabling precise cropping for further analysis or manipulation.

## Advantages of Object Cropping

- **Focused Analysis:** VajraV1 facilitates targeted object cropping, allowing for in-depth examination or processing of individual items within a scene.
- **Reduced Data Volume:** By extracting only relevant objects, object cropping helps in minimizing data size, making it efficient for storage, transmission or subsequent computational tasks.
- **Enhanced Precision:** VajraV1's object detection accuracy ensures that the cropped objects maintain their spatial relationships, preserving the integrity of the visual information for detailed analysis.

## Examples:

!!! example "Object Cropping using Vayuvahana Technologies VayuAI"

    === "CLI"

        ```bash
        # Crop the objects
        vajra solutions crop show=True

        # Pass a source video
        vajra solutions crop source="path/to/video.mp4"

        # Crop specific classes
        vajra solutions crop classes="[0, 2]"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Initialize object cropper object
        cropper = solutions.ObjectCropper(
            show=True,  # display the output
            model="vajra-v1-nano-det.pt",  # model for object cropping
            classes=[0, 2],  # crop specific classes i.e. person and car with COCO pretrained model.
            # conf=0.5,  # adjust confidence threshold for the objects.
            # crop_dir="cropped-detections",  # set the directory name for cropped detections
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = cropper(im0)

            # print(results)  # access the output

        cap.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

### `ObjectCropper` Arguments

Here's a table with the `ObjectCropper` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "crop_dir"]) }}

Moreover, the following visualization arguments are available for use:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width"]) }}

## FAQ

### What is object cropping in the Vayuvahana Technologies VayuAI and how does it work?

Object cropping using [Vayuvahana Technologies VayuAI](https://github.com/NamanMakkar/VayuAI) involves isolating and extracting specific objects from an image or video based on VajraV1's detection capabilities. This process allows for focused analysis, reduced data volume, and enhanced precision by leveraging VajraV1 to identify objects with high accuracy and crop them accordingly. For an in-depth tutorial, refer to the [object cropping example](#object-cropping-using-vayuvahana-technologies-vayuai).

### Why should I use Vayuvahana Technologies VayuAI's VajraV1 for object cropping over other solutions?

VayuAI's VajraV1 stands out due to its precision, speed, accuracy and ease of use. It allows detailed and accurate object detection and cropping, essential for [focused analysis](#advantages-of-object-cropping) and applications needing high data integrity. Moreover, VajraV1 integrates seamlessly with tools like OpenVINO and TensorRT for deployments requiring real-time capabilities and optimization on diverse hardware. Explore the benefits in the [guide on model export](../modes/export.md).

### How can I reduce the data volume of my dataset using object cropping?

By using VajraV1 to crop only relevant objects from your images or videos, you can significantly reduce the data size, making it more efficient for storage and processing. This process involves training the model to detect specific objects and then using the results to crop and save these portions only.

### What are the hardware requirements for efficiently running VajraV1 for object cropping?

Vayuvahana Technologies VayuAI's VajraV1 is optimized for both CPU and GPU environments, but to achieve optimal performance, especially for real-time or high-volume inference, a dedicated GPU (e.g. NVIDIA Tesla, RTX series) is recommended. For deployment on lightweight devices, consider using CoreML for iOS or TFLite for Android. More details can be found in the guide on [model deployment options](../guides/model-deployment-options.md).