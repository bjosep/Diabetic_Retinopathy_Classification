import numpy as np
import streamlit as st
from PIL import Image
from utils import process_image, get_footer, load_model
from copy import deepcopy

def highlight_prediction(options, idx):
    options = deepcopy(options)
    highlight = f'''<span style="color:green">**{options[idx]}** </span>'''
    options[idx] = highlight
    return '<br>'.join(options)


if __name__ == '__main__':
    st.image("assets/ophtalmo_img.jpg")
    st.markdown("<h1 style='text-align: center; color: black;'>"
                "<center>&emsp;&emsp;Detection of Diabetic Retinopathy from Medical Images</center></h1>", unsafe_allow_html=True)
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a retina medical image...", type=["jpg","png"])
    st.write(" ")
    st.markdown(get_footer(), unsafe_allow_html=True)

    if uploaded_file is not None:

        options = ['No Diabetic Retinopathy', 'Mild', 'Moderate' 'Severe',
                   'Profelivative Diabetic Retinopathy']
        img_in = Image.open(uploaded_file)
        img_in_processed = process_image(img_in)

        col1, col2 = st.columns(2)
        col1.image(img_in_processed)
        st.write("")

        model = load_model('assets/model_2021-08-30')
        prediction = model.predict(img_in_processed).ravel()
        idx = np.argmax(prediction)
        col2.markdown("### Severity of Diabetic Retinopathy")
        col2.markdown(highlight_prediction(options, idx), unsafe_allow_html=True)

