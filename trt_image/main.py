import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Function to preprocess the image for MRI model
def preprocess_mri_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((128, 128))  # Resize to 128x128
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=-1) # Add channel dimension for grayscale
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to preprocess the image for X-ray model
def preprocess_xray_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((128, 128))  # Resize to 128x128
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=-1) # Add channel dimension for grayscale
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def load_models():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mri_model_path = os.path.join(script_dir, 'advanced_tumor_classification_model.h5')
    xray_model_path = os.path.join(script_dir, 'chest_xray_model.h5')

    try:
        mri_model = tf.keras.models.load_model(mri_model_path)
        st.success("MRI model loaded successfully!")
        xray_model = tf.keras.models.load_model(xray_model_path)
        st.success("X-ray model loaded successfully!")
        return mri_model, xray_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error("Please ensure the model files ('advanced_tumor_classification_model.h5' and 'chest_xray_model.h5') are in the same directory as the script and are valid Keras model files.")
        st.stop()
        return None, None

def main():
    
    mri_model, xray_model = load_models()

    if mri_model is None or xray_model is None:
        return # Stop execution if models failed to load

    st.title("Image Analysis")

    analysis_type = st.selectbox("Choose analysis type:", ["MRI Tumor Analysis", "X-ray Image Analysis"])
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze"):
            if analysis_type == "MRI Tumor Analysis":
                processed_image = preprocess_mri_image(image)
                prediction = mri_model.predict(processed_image)
                
                mri_class_labels = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
                
                predicted_class_index = np.argmax(prediction[0])
                predicted_class_label = mri_class_labels[predicted_class_index]
                confidence = prediction[0][predicted_class_index]
                
                st.write(f"Prediction: {predicted_class_label}")
                st.write(f"Confidence: {confidence:.2f}")

            elif analysis_type == "X-ray Image Analysis":
                processed_image = preprocess_xray_image(image)
                prediction = xray_model.predict(processed_image)
                
                # Assuming the X-ray model is a binary classifier (Normal/Pneumonia)
                # Adjust if it's multi-class or if the interpretation of output is different
                if prediction[0][0] > 0.5: 
                    st.write("Prediction: Pneumonia Detected")
                else:
                    st.write("Prediction: Normal")
                st.write(f"Confidence: {prediction[0][0]:.2f}")

if __name__ == "__main__":
    main()
