import streamlit as st
import pandas as pd
import numpy as np

st.title("Biomaterial Electrode Optimization AI")
st.write("Upload an Excel file with a sheet named 'Master_Dataset', or run the app where the file exists.")

uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"]) 

df = None
if uploaded is not None:
    try:
        df = pd.read_excel(uploaded, sheet_name="Master_Dataset")
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
else:
    # Uncomment and adjust the path below if you want the app to load a local file when run on your machine/server
    # local_path = r"C:\Users\hp\Documents\GitHub\my projects\Data\Biomaterial_Electrode_Literature_Dataset_REVISED.xlsx"
    # try:
    #     df = pd.read_excel(local_path, sheet_name="Master_Dataset")
    # except Exception as e:
    #     st.info("No upload and could not read local file: " + str(e))
    pass

if df is not None:
    df.replace("NR", pd.NA, inplace=True)
    st.write(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    st.dataframe(df.head())
    
    # Example: show basic column types
    st.write("Column summary:")
    st.write(df.dtypes)
else:
    st.info("No DataFrame loaded yet. Upload a file or enable a local path in the script.")
