import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
from utils.predict import predict_address_onnx
from utils.standardize import return_cannon

model_path = "./model_files/model_quantized_onnx/model.onnx"
tokenizer_path = "./model_files/model_quantized_onnx"

st.set_page_config(
    layout="wide",
    page_title="address-resolution-demo",
)

st.title("AI-Powered Address Resolution & Standardization App")

st.subheader("Enter an address below to see it parsed, normalized, and labeled.")

text_input = st.text_input(
        label="Write the address here:",
        key="placeholder",
        width="stretch"
    )

if text_input != "":
    ner_output, table = return_cannon(address=text_input)
    prediction = predict_address_onnx(text_input, model_path, tokenizer_path)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Address Parsing & Standardization")
        st.write("---")
        
        st.subheader("NER Model Output")
        st.write(f"**City:** {ner_output['il']}")
        st.write(f"**District:** {ner_output['ilce']}")
        st.write(f"**Quarter:** {ner_output['mahalle']}")
        st.write(f"**Street:** {ner_output['sokak']}")
        st.write(f"**Additional Info:** {ner_output['diger']}")
        
    with col2:
        st.header("Prediction Results")
        st.write("---") 
        
        st.subheader(f"Label: {prediction}")
        st.write("### Best Overall Score Address:")
        st.write(f"**City:** {table['city_name'][0]}")
        st.write(f"**District:** {table['district_name'][0]}")
        st.write(f"**Quarter:** {table['quarter_name'][0]}")
        st.write(f"**Street:** {table['street_name'][0]}")
        st.write(f"**Additional Info:** {ner_output['diger']}")

    st.header("Best Matches")
    st.write("---")
    st.table(table)
