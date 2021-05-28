import streamlit as st
import json
import torch
from collections import Counter

import generate_text


with open('model/word_to_id.json') as json_file:
    word_to_id = Counter(json.load(json_file))


id_to_word = ["<Unknown>"] + [word for word, index in word_to_id.items()]

net = torch.load('model/trained_model.pt')
net.eval()

st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0.2)), url("https://data.whicdn.com/images/328275119/original.gif");
        background-size: contain;
        background-blend-mode: darken
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title('Restaurant Review Generator')

word = st.text_input(
    "What do you want to know?", value=""
)

def val_style(text):
    html_temp = f"""
    <div style = "background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)); padding: 10px 10px 10px 10px; border-radius: 20px; "> 
    <p style = " font: color:white; text_align:center;"> {text} </p>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)

if st.button("Generate"):
    if (word == ""):
        st.title("please enter keyword!")
    else:
        generated_text = generate_text.prediction(net, word_to_id, id_to_word, word, 10, 5)
        val_style(generated_text)
   
    
    
    
    
    
    
