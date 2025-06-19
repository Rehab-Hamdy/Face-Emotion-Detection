import streamlit as st
import numpy as np
import cv2
from mtcnn import MTCNN
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Load your model and class names
model = keras.models.load_model('cnn_model.h5')
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Image Preprocessing
def preprocess_fer2013(img):
    
    img = img.astype(np.uint8)

    # Noise Reduction (should be done before the Contrast Adjustment)
    img = cv2.fastNlMeansDenoising( # Fast - Non-local Means Denoising
        img, None,
        h=7,   # Filter strength (5-10) (higher = more denoising) Higher values remove more noise but may lose some details
        templateWindowSize=5, # Size of the patch used to compare similarity (5x5 pixels) | Larger patches capture more structural information but are computationally expensive
        searchWindowSize=15   # Region area around each pixel where similar patches are searched (15x15 pixels) | Larger windows find more similar patches but increase computation time
    )

    # Contrast Limited Adjustment Histogram Equalization (CLAHE) -> Can help make facial features more distinguishable (improves contrast locally)
    clahe = cv2.createCLAHE(
        clipLimit=2.0, # clipLimit : Limits the amplification of contrast to prevent noise over-enhancement.
        tileGridSize=(8, 8) # Divides the image into an 8x8 grid of tiles (64 tiles total). CLAHE processes each tile independently.
    )
    img = clahe.apply(img)

    # Combines the original and blurred image to enhance edges
    blurred = cv2.GaussianBlur( # Apply Gaussian Blurring on the image
        img, # input image
        (3, 3), # kernel 3 x 3
        1.0 # Sigma of the Gaussian
    )

    # Sharpens the image by subtracting a blurred version (Unsharp Masking)
    img = cv2.addWeighted( # result = (1.2 * original) + (-0.2 * blurred) + 0
        img, # The original image
        1.2, # weight i multiply in the original one
        blurred, # the blurred image after applying Gaussian 
        -0.2, # weight i multiply in the blurred image
        0
    ) 
    
    img = img / 255.0 # Normalization -> Rescale pixel values to [0, 1]
    
    img = np.expand_dims(img, axis=-1)
    # Add channel dimension -> Shape: (48, 48, 1) For the input shape of CNN. 1 : For one channel (grayscal)
    # Shape: (48, 48, 3) For RGB image => 3 channels

    return img


# Streamlit UI
st.set_page_config(page_title="Emotion Detector", layout="centered")
st.title("😊 Facial Emotion Recognition")

uploaded_file = st.file_uploader("Upload an image with a face", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)  # <-- Now works (PIL imported)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image.convert("RGB"))
    detector = MTCNN()
    faces = detector.detect_faces(img)

    if faces:
        for face in faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cropped = img[y:y+h, x:x+w]
            resized = cv2.resize(cropped, (48, 48))
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            processed = preprocess_fer2013(gray)
            processed = np.expand_dims(processed, axis=0)
            prediction = model.predict(processed)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            label = f"{predicted_class} ({confidence:.2f}%)"

            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1, cv2.LINE_AA)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Image with Detected Faces", use_column_width=True)
    else:
        st.warning("No face detected in the image.")