import streamlit as st
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing import image
from PIL import Image

# Load Model
model = tf.keras.models.load_model(
    "cat_dog_mobilenet.keras"
)

# Page Title
st.title("Cat vs Dog Classifier")

st.write("Upload a Cat or Dog image")

# Upload Image
uploaded_file = st.file_uploader(

    "Choose an image",

    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open Image
    img = Image.open(uploaded_file)

    # Display Image
    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize Image
    img = img.resize((224,224))

    # Convert To Array
    img_array = image.img_to_array(img)

    # Normalize
    img_array = img_array / 255.0

    # Expand Dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    classes = ["Cat", "Dog"]

    result = classes[predicted_class]

    # Output
    st.subheader(f"Prediction: {result}")

    st.write(f"Confidence: {confidence:.2f}%")