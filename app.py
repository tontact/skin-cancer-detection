import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras import preprocessing
from keras.models import load_model
from keras.activations import sigmoid
import os
import h5py

st.title(':blue[MLFlow Prediction App]')
st.header('Skin Cancer Prediction')
st.text("Upload a skin cancer Image for image classification")

def main():
    file_uploaded = st.file_uploader('Choose the file', type=['jpg', 'png', 'jpeg'])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result = predict_class(image)
        st.write(result)
        st.pyplot(figure)

def predict_class(image):
    classifier_model = tf.keras.models.load_model('skin_detect_model.h5')
    # shape = ((75, 100, 3))  # Expected input shape of the model
    test_image = image.resize((3, 3))  
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = np.expand_dims(test_image, axis=3)  # Expand dims to match (batch_size, height, width, channels, filters)
    test_image = np.repeat(test_image, 32, axis=-1)  # Repeat the image along the filters dimension

    class_names = ['actinic keratosis',
                   'basal cell carcinoma',
                   'dermatofibroma',
                   'melanoma',
                   'nevus',
                   'pigmented benign keratosis',
                   'seborrheic keratosis',
                   'squamous cell carcinoma',
                   'vascular lesion']

    predictions = classifier_model.predict(test_image)
    predictions = tf.where(predictions < 0.5, 0, 1)
    scores = predictions.numpy()
    image_class = class_names[np.argmax(scores)]
    result = 'The image predicted is : {}'.format(image_class)
    return result

if __name__ == "__main__":
    main()
