# Image Segmentation App

This is a simple Tkinter-based application for loading images and performing image segmentation using a pre-trained model.

## Clone the Repository

To clone this repository and run the application locally, use the following commands:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

## Prerequisites

Before running the application, make sure you have the required dependencies installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```

## Usage
Run the application:

```bash
python testing_model.py
```

Load an image by clicking the "Load Image" button.

Perform image segmentation by clicking the "Segment Image" button.

![result](https://github.com/InnaK342/Airbus_Ship_Detection_Inna/tree/main/images/result.jpg)

## Dependencies
1. Pillow: Python Imaging Library (Fork)
2. NumPy: Fundamental package for scientific computing with Python
3. OpenCV: Open Source Computer Vision Library
4. TensorFlow: Open-source machine learning framework

## Notes
The application uses a pre-trained model (final_model.h5) for image segmentation. Make sure the model file is available in the same directory as the script.

The image segmentation result is displayed using the OpenCV library.

