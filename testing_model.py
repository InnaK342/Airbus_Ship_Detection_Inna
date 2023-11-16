import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
loaded_model = tf.keras.models.load_model('final_model.h5', compile=False)

def preprocess_image(image, img_scaling):
    # Apply scaling and color conversion
    image = image[::img_scaling[0], ::img_scaling[1]]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize the image
    image = image / 255.0

    # Add batch dimension
    image = tf.expand_dims(image, axis=0)

    return image

def postprocess_prediction(prediction):
    # Squeeze the prediction to remove the batch dimension
    prediction = np.squeeze(prediction, axis=0)

    # Add any post-processing steps here if needed

    return prediction

class ImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Segmentation App")
        self.root.geometry('500x500')

        # Button for loading an image
        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack(padx=30, pady=10)

        # Button for performing image segmentation
        self.segment_button = tk.Button(root, text="Segment Image", command=self.segment_image)
        self.segment_button.pack(padx=30, pady=10)

        # Widget for displaying the image
        self.image_label = tk.Label(root)
        self.image_label.pack(padx=10, pady=5)

        # Current image and its path
        self.current_image = None
        self.image_path = None

    def load_image(self):
        # Open a file dialog to choose a file
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif")])

        if file_path:
            # Open and display the image in Tkinter
            image = Image.open(file_path)
            image = image.resize((400, 400))
            tk_image = ImageTk.PhotoImage(image)
            self.image_label.config(image=tk_image)
            self.image_label.image = tk_image

            # Save the current image and its path
            self.current_image = cv2.imread(file_path)
            self.image_path = file_path

    def segment_image(self):
        if self.current_image is not None:
            # Expand the dimensions of the image (add batch dimension)
            input_image = preprocess_image(self.current_image, (3, 3))

            # Apply the model to the image
            result = loaded_model.predict(input_image)
            prediction = postprocess_prediction(result)

            # Convert the prediction to a color image
            color_image = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)

            # Display the segmented image using OpenCV
            cv2.imshow('Segmented Image', color_image)
        else:
            messagebox.showinfo("Info", "Please load an image first.")


root = tk.Tk()
app = ImageSegmentationApp(root)
root.mainloop()
