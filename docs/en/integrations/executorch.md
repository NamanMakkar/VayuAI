# Deploy VajraV1 on Edge Devices with ExecuTorch

## Usage

!!! example "Usage"

    === "Python"

        ```python
        from vajra import Vajra

        model = Vajra("vajra-v1-nano-det.pt")
        model.export(format="executorch")

        executorch_model = Vajra("vajra-v1-nano-det_executorch_model")
        results = executorch_model.predict("path/to/img.jpg")

        ```

    === "CLI"

        ```bash

        vajra export model=vajra-v1-nano-det.pt format=executorch

        vajra predict model=vajra-v1-nano-det_executorch_model source="path/to/img.jpg"

        ```

## Export Arguments

When exporting to ExecuTorch format, you can specify the following arguments:

| Argument | Type            | Default | Description                                |
| -------- | --------------- | ------- | ------------------------------------------ |
| `img_size`  | `int` or `list` | `640`   | Image size for model input (height, width) |
| `device` | `str`           | `'cpu'` | Device to use for export (`'cpu'`)         |


## Output Structure

The ExecuTorch export creates a directory containing the model and metadata:

```text
vajra-v1-nano-det_executorch_model/
├── vajra-v1-nano-det.pte              # ExecuTorch model file
└── metadata.yaml            # Model metadata (classes, image size, etc.)
```

## Mobile Integration

Example iOS Integration (Objective-C/C++):

```objc
#include <executorch/extension/module/module.h>

using namespace ::executorch::extension;

Module module("/path/to/vajra-v1-nano-det.pte");

float input[1 * 3 * 640 * 640];
auto tensor = from_blob(input, {1, 3, 640, 640});

const auto result = module.forward(tensor);
```

Example Android Integration (Kotlin)

```kotlin
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

val module = Module.load("/path/to/vajra-v1-nano-det.pte")

val inputTensor = Tensor.fromBlob(floatData, longArrayOf(1, 3, 640, 640))
val inputEValue = EValue.from(inputTensor)

val outputs = module.forward(inputEValue)
val scores = outputs[0].toTensor().dataAsFloatArray
```

## Embedded Linux

For embedded Linux systems, use the ExecuTorch C++ API:

```cpp
#include <executorch/extension/module/module.h>

auto module = torch::executor::Module("vajra-v1-nano-det.pte");

std::vector<float> input_data = preprocessImage(image);
auto input_tensor = torch::executor::Tensor(input_data, {1, 3, 640, 640});

auto outputs = module.forward({input_tensor});
```
For more details, visit the [ExecuTorch Documentation](https://docs.pytorch.org/executorch/)

## Summary

Exporting VajraV1 models to ExecuTorch format enables efficient deployment on mobile and edge devices. With PyTorch-native integration, cross-platform support, and optimized performance, ExecuTorch is an excellent choice for edge AI applications.

Key takeaways:

- ExecuTorch provides PyTorch-native edge deployment with excellent performance
- Export is simple with `format='executorch'` parameter
- Models are optimized for mobile CPUs via XNNPACK backend
- Supports iOS, Android, and embedded Linux platforms
- Requires Python 3.10+ and FlatBuffers compiler


## FAQ

### How do I export a VajraV1 model to ExecuTorch?

Export a VajraV1 model to ExecuTorch using either Python or CLI:

```python
from vajra import Vajra

model = Vajra("vajra-v1-nano-det.pt")
model.export(format="executorch")
```

or

```bash
vajra export model=vajra-v1-nano-det.pt format=executorch
```

### Can I run inference with ExecuTorch models directly in Python?

ExecuTorch models (`.pte` files) are designed for deployment on mobile and edge devices using the ExecuTorch runtime. They cannot be directly loaded with `Vajra()` for inference in Python. You need to integrate them into your target application using the ExecuTorch runtime libraries.

### What platforms are supported by ExecuTorch?

ExecuTorch supports:

- **Mobile**: iOS and Android
- **Embedded Linux**: Raspberry Pi, NVIDIA Jetson, and other ARM devices
- **Desktop**: Linux, macOS, and Windows (for development)

### How does ExecuTorch compare to TFLite for mobile deployment?

Both ExecuTorch and TFLite are excellent for mobile deployment:

- **ExecuTorch**: Better PyTorch integration, native PyTorch workflow, growing ecosystem
- **TFLite**: More mature, wider hardware support, more deployment examples

Choose ExecuTorch if you're already using PyTorch and want a native deployment path. Choose TFLite for maximum compatibility and mature tooling.

### Can I use ExecuTorch models with GPU acceleration?

Yes! ExecuTorch supports hardware acceleration through various backends:

- **Mobile GPU**: Via Vulkan, Metal, or OpenCL delegates
- **NPU/DSP**: Via platform-specific delegates
- **Default**: XNNPACK for optimized CPU inference

Refer to the [ExecuTorch Documentation](https://docs.pytorch.org/executorch/stable/compiler-delegate-and-partitioner.html) for backend-specific setup.
