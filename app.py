import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

st.title("📈 Stock Market Analysis Dashboard")

folder_path = "D:\\DA\\PROJECTS\\Stock Market Analysis + Prediction\\Data"   # keep stocks folder in same directory

# all_data = []

# for file in os.listdir(folder_path):
#     if file.endswith(".csv"):
#         df = pd.read_csv(os.path.join(folder_path, file))
#         df['Symbol'] = file.replace(".csv", "")
#         all_data.append(df)

# final_df = pd.concat(all_data)
# final_df['Date'] = pd.to_datetime(final_df['Date'])

# pivot_df = final_df.pivot(index='Date', columns='Symbol', values='Close')

# # Dropdown
# stock = st.selectbox("Select Stock", pivot_df.columns)

# # Plot
# fig, ax = plt.subplots()
# ax.plot(pivot_df[stock])
# ax.set_title(f"{stock} Price Trend")

# st.pyplot(fig)

files = os.listdir(folder_path)

stock = st.selectbox("Select Stock", files)

df = pd.read_csv(os.path.join(folder_path, stock))

plt.plot(df['Close'])
plt.title(stock)

st.pyplot(plt)