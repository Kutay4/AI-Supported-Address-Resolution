import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
from utils.predict import predict_address_onnx

model_path = "model_quantized_onnx/model.onnx"
tokenizer_path = "model_quantized_onnx"

st.set_page_config(
    layout="wide",
    page_title="address-resolution-demo",
)

st.title("AI-Powered Address Resolution & Standardization App")

st.columns((2,1))

col1, col2 = st.columns(2)

with col1:
    st.markdown("# Enter an address below to see it parsed, normalized, and labeled.")

    text_input = st.text_input(
            label="Write the address",
            key="placeholder",
            width="stretch"
        )

with col2:
    if text_input != "":
        st.markdown(f"# Label: {predict_address_onnx(text_input, model_path, tokenizer_path)}")

