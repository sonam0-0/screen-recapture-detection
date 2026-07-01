# Screen Recapture Detection using MobileNetV2

A lightweight deep learning project that classifies images as **Real** or **Screen Recapture** using **MobileNetV2** and transfer learning.

## Features

* Binary image classification (Real vs Screen)
* MobileNetV2-based transfer learning
* Fast image prediction
* OpenCV image preprocessing
* TensorFlow/Keras implementation

## Tech Stack

* Python
* TensorFlow / Keras
* MobileNetV2
* OpenCV
* NumPy

## Dataset

* **Real:** Camera-captured images
* **Screen:** Screenshots or photos of digital screens

## Model Performance

| Metric              |       Value |
| ------------------- | ----------: |
| Training Accuracy   |   **97.7%** |
| Validation Accuracy |   **95.8%** |
| Model               | MobileNetV2 |

## Usage

Train the model:

```bash
python train.py
```

Predict an image:

```bash
python predict.py path/to/image.jpg
```

## Future Work

* Web application deployment
* Real-time webcam detection
* Grad-CAM visualization
* TensorFlow Lite support
