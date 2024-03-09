import io
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st

# st.title(':blue[MLFlow Prediction App]')
# st.header('Skin Cancer Prediction')
# st.text("Upload a skin cancer Image for image classification")
# Load the model
model = tf.keras.models.load_model('skin_detect_model.h5')

def preprocess_image(uploaded_image):
    resized_image = uploaded_image.resize((100, 75))
    image_array = np.array(resized_image)
    image_array = image_array / 255.
    return image_array

def prediction(image_array):
    pred = model.predict(np.expand_dims(image_array, axis=0))
    return pred

def main():
    st.markdown(":blue[Prediction App]")
    st.header('Skin Cancer Classification')
    st.text("Upload a skin Image for image classification")

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        image_array = preprocess_image(image)

        # Make predictions
        ans = prediction(image_array)
        classes = ['Melanocytic nevi',
                   'Melanoma',
                   'Benign keratosis-like lesions',
                   'Basal cell carcinoma',
                   'Actinic keratoses',
                   'Vascular lesions',
                   'Dermatofibroma']

        st.write("Prediction probabilities:")
        for i in range(len(classes)):
            st.write(f"Class Name: {classes[i]} ({ans[0][i]})")

        st.write("")  # Add a space after printing the prediction probabilities
        result = 'The image predicted is : {}'.format(classes[np.argmax(ans)])
        st.write(result)

if __name__ == "__main__":
    main()
