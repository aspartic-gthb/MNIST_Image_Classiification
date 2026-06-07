import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import cv2

# Set page configuration
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="wide"
)

st.title("🔢 Handwritten Digit Recognition")
st.markdown("Draw a digit (0-9) in the canvas below, and the deep CNN model will classify it instantly.")

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras")

model = load_model()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Draw Here")
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="black",  # Fixed fill color with some opacity
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        update_streamlit=True,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("Prediction")
    if canvas_result.image_data is not None:
        # Get the image data from canvas (RGBA)
        img = canvas_result.image_data
        
        # Convert RGBA to Grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        
        # Check if the canvas is empty (all pixels are black)
        if np.any(img_gray):
            # Resize to 28x28
            img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
            
            # Show the resized input to user
            st.image(img_resized, caption="28x28 Input", width=100)
            
            # Normalize the image
            img_normalized = img_resized.astype("float32") / 255.0
            
            # Z-score normalization as trained
            mean = 0.1307
            std = 0.3081
            img_normalized = (img_normalized - mean) / std
            
            # Expand dimensions to match model input (1, 28, 28, 1)
            img_tensor = np.expand_dims(img_normalized, axis=(0, -1))
            
            # Predict
            predictions = model.predict(img_tensor)[0]
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class]
            
            st.markdown(f"### Predicted Digit: **{predicted_class}**")
            st.markdown(f"### Confidence: **{confidence * 100:.2f}%**")
            
            st.write("---")
            st.write("**Top 3 Predictions:**")
            top_3_indices = np.argsort(predictions)[::-1][:3]
            for idx in top_3_indices:
                st.write(f"- Digit {idx}: {predictions[idx] * 100:.2f}%")
        else:
            st.info("Draw a digit on the canvas to see the prediction.")

st.markdown("---")
st.markdown("Built with [Streamlit](https://streamlit.io/) and TensorFlow.")
