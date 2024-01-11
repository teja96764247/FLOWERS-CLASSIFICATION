import streamlit as st
import tensorflow as tf

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache_data
def load_model():
    model=tf.keras.models.load_model('my_model.hdf5')
    return model
model=load_model()

st.write("""
       #Flower Classification web app
""")

file=st.file_uploader("Please upload an flower image",type=['jpg','png'])

import cv2
from PIL import Image,ImageOps
import numpy as np

def predict_function(img,model):
    size=(64,64)
    image=ImageOps.fit(img,size,Image.ANTIALIAS)
    img_arr=np.asarray(image)
    img_scaled=img_arr/255
    img_reshape=np.reshape(img_scaled,[1,64,64,3])
    prediction=model.predict(img_reshape)
    output=np.argmax(prediction)
    if(output==1):
        return "The Flower name is DAISY"
    elif(output==2):
        return "The Flower name is DANDELION"
    elif(output==3):
        return "The Flower name is TULIP"
    elif(output==4):
        return "The Flower name is SUNFLOWER"
    elif(output==5):
        return "The Flower name is ROSE"
    else:
        return "SORRY we did not found the name os the Flower"

    

if file is None:
    st.text('Please upload an image file')
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    result=predict_function(image,model)
    st.success(result)