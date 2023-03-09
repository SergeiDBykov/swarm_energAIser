import streamlit as st
import time
from PIL import Image
import base64
import os

curent_path = os.path.dirname(os.path.abspath(__file__))

logo = Image.open(curent_path +"\logo.jpg")
st.set_page_config(page_title="EnergAIser", page_icon=logo, layout="centered", initial_sidebar_state="auto", menu_items=None)

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background(curent_path +r"\background.jpg")
