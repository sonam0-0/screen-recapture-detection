# Screen Recapture Detection using MobileNetV2

A deep learning-based image classification system that detects whether an input image is a **Real Photograph** or a **Screen Recapture** (such as a screenshot or a photo taken of a digital display). The model is built using **MobileNetV2** with transfer learning, providing high accuracy while remaining lightweight and fast for inference.



## Features

* Binary image classification (Real vs Screen)
* Transfer Learning using **MobileNetV2**
* Fast and lightweight deep learning model
* Image preprocessing using OpenCV
* Command-line prediction interface
* TensorFlow/Keras implementation
* Efficient inference with confidence score


## Technologies Used

* Python 3.x
* TensorFlow / Keras
* **MobileNetV2**
* OpenCV
* NumPy
* Matplotlib



##  Project Structure

```text
Screen-Recapture-Detection/
│
├── dataset/
│   ├── real/
│   └── screen/
├── predict.py
├── requirements.txt
├── README.md
└── ...
```



##  Dataset

The dataset contains two image categories:

* **Real** – Images captured directly from a camera.
* **Screen** – Images that are screenshots or photographs of digital screens.

The images are automatically preprocessed before training.



##  Model Architecture

This project uses **MobileNetV2**, a lightweight convolutional neural network designed for efficient image classification.

### Training Pipeline

1. Load image dataset
2. Resize images to the required input size
3. Normalize pixel values
4. Apply MobileNetV2 as the base model
5. Add custom classification layers
6. Train the model
7. Save the trained model
8. Predict unseen images



##  Model Performance

| Metric              | Value                |
| ------------------- | -------------------- |
| Training Accuracy   | **97.7%**            |
| Validation Accuracy | **95.8%**            |
| Number of Classes   | 2                    |
| Model               | MobileNetV2          |
| Loss Function       | Binary Cross-Entropy |
| Optimizer           | Adam                 |

> **Note:** Replace the accuracy values above with your actual training results if they are different.



##  Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/screen-recapture-detection.git
cd screen-recapture-detection
```

Install the required packages:

```bash
pip install -r requirements.txt
```



##  Train the Model

Run:

```bash
python train.py
```

The trained model will be saved automatically in the `model/` directory.



##  Predict an Image

Run:

```bash
python predict.py path/to/image.jpg
```

Example:

```bash
python predict.py dataset/real/sample.jpg
```

### Sample Output

```text
Confidence : 0.98
Prediction : REAL
Latency    : 32 ms
```



##  Future Improvements

* Web application using Flask or Streamlit
* Real-time webcam detection
* Grad-CAM visualization for explainability
* Batch image prediction
* Larger and more diverse dataset
* Mobile deployment using TensorFlow Lite




