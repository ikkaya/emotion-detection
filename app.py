import streamlit as st
import numpy as np
import pandas as pd
from fer import FER
from PIL import Image
import cv2

@st.cache
def getEmotions(img):
    detector  = FER(mtcnn=True)
    result = detector.detect_emotions(img)
    data  = result[0]['emotions']
    if data is None:
        st.write('No result')
        return False
    else:
        return data

html_temp = """
    <body>
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Emotion Detection WebApp</h2>
    </div>
    </body>
    """
st.markdown(html_temp, unsafe_allow_html=True)
st.subheader('This is an app to return emotions of image')


file = st.sidebar.file_uploader('Please upload an image file', type = ['jpg', 'jpeg', 'png'])

if file is None:
    st.write("You haven't uploaded an image file")
else:
    image = Image.open(file)
    img = np.array(image)
    st.image(image, use_column_width=True)
    st.write(pd.DataFrame(getEmotions(img), index=[0]))
    st.write("The main emotion was detected as",pd.DataFrame(getEmotions(img), index=[0]).idxmax(axis="columns").iloc[0],".")
    prediction = pd.DataFrame(getEmotions(img), index=[0]).idxmax(axis="columns").iloc[0]
    if prediction == 'angry':
        st.subheader("You seem to be Angry :rage: today. Take it easy! ")
    elif prediction == 'disgust':
        st.subheader("You seem to be Disgust :rage: today! ")
    elif prediction == 'fear':
        st.subheader("You seem to be Fearful :fearful: today. Don't lose yourself in your fear! ")
    elif prediction == 'happy':
        st.subheader("Yeah!  You are Happy :smile: today. Make everyday a happy day! ")
    elif prediction == 'sad':
        st.subheader("You seem to be Sad :sad: today. Smile and be happy! ")
    elif prediction == 'Surprise':
        st.subheader("You seem to be Surprised today! ")
    elif prediction == 'neutral':
        st.subheader("You seem to be Neutral today. Wish you a happy day! ")
    else:
        st.write("OK!")


