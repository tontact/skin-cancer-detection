import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
# Load the model
model = tf.keras.models.load_model('skin_detect_model.h5')
model.summary()

def preprocess_image(uploaded_image):
    resized_image = uploaded_image.resize((100, 75))  # Resize image to match model input shape
    image_array = np.array(resized_image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def prediction(image_array):
    pred = model.predict(np.expand_dims(image_array, axis=0))
    return pred

def main():
    st.title('Skin Lesion Classifier')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
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

        # Print prediction results
        st.subheader("Prediction Probabilities:")
        for i in range(len(classes)):
            st.write(f"{classes[i]}: {ans[0][i]}")

        result = 'The image predicted is : {}'.format(classes[np.argmax(ans)])
        st.success(result)

if __name__ == "__main__":
    main()
