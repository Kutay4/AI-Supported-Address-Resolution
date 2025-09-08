import streamlit as st
import pandas as pd
from utils.predict import predict_address_onnx, predict_csv_onnx
from utils.standardize import return_cannon
import time

model_path = "./model_files/model_quantized_onnx/model.onnx"
tokenizer_path = "./model_files/model_quantized_onnx"

st.set_page_config(
    layout="wide",
    page_title="address-resolution-demo",
)

st.title("AI-Powered Address Resolution & Standardization App")

st.subheader("Enter an address below to see it parsed, normalized, and labeled.")

text_input = st.text_input(
    label="Write the address here:", key="placeholder", width="stretch"
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

st.title("Inference from CSV file")
st.subheader("Upload a CSV file and get predictions from the model.")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type="csv",
    help="Upload your CSV file with address string for prediction.",
)

if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file, index_col=0)

        with st.expander("Show Uploaded Dataset"):
            st.dataframe(df_uploaded)

        st.write("---")
        if st.button("Start Inference"):
            with st.spinner("Processing your data..."):
                progress_bar = st.progress(0, text="Inference in progress...")

                start_time = time.perf_counter()
                df_with_predictions = predict_csv_onnx(
                    df_uploaded, model_path, tokenizer_path, progress_bar
                )

                end_time = time.perf_counter()
                elapsed_time = end_time - start_time

                st.success(f"Inference completed in {elapsed_time:.4f} seconds.")
                progress_bar = st.progress(0, text="Adress Parsing in progress...")

                start_time = time.perf_counter()

                rows = []
                for t, address in enumerate(df_with_predictions["address"]):
                    progress_percentage = (t + 1) / len(df_with_predictions)
                    progress_bar.progress(progress_percentage, text=f"{(t + 1)}%")
                    _, table = return_cannon(address=address, return_top=1)
                    rows.append(
                        table[
                            "city_name", "district_name", "quarter_name", "street_name"
                        ].to_dicts()[0]
                    )

                ner_output_df = pd.DataFrame(rows)
                df_with_predictions = df_with_predictions.merge(
                    ner_output_df,
                    left_index=True,
                    right_index=True,
                )
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                st.success(f"Address Parsing completed in {elapsed_time:.4f} seconds.")
            st.subheader("Prediction Results")
            st.dataframe(df_with_predictions)

            st.download_button(
                label="Download Predictions as CSV",
                data=df_with_predictions.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv",
                icon="⬇️",
            )

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
