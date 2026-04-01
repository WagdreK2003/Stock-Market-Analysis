import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

st.title("📈 Stock Market Analysis Dashboard")

folder_path = "Data"   # keep stocks folder in same directory

files = os.listdir(folder_path)

stock = st.selectbox("Select Stock", files)

df = pd.read_csv(os.path.join(folder_path, stock))

plt.plot(df['Close'])
plt.title(stock)

st.pyplot(plt)