# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import kagglehub

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(page_title="Income Prediction App", layout="centered")

st.title("💰 Income Clustering using K-Means")
st.write("Enter **Age** to see which **Income group (cluster)** it belongs to")

# ---------------------------------
# Load Dataset
# ---------------------------------
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("duajanmuhammed/kmean-data")
    files = os.listdir(path)
    csv_file = [f for f in files if f.endswith(".csv")][0]
    return pd.read_csv(os.path.join(path, csv_file))

df = load_data()

# ---------------------------------
# Train Model
# ---------------------------------
X = df[['Age', 'Income($)']]

kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# ---------------------------------
# User Input
# ---------------------------------
st.subheader("🔢 Enter Age")

age_input = st.number_input(
    "Age",
    min_value=int(df['Age'].min()),
    max_value=int(df['Age'].max()),
    value=int(df['Age'].mean())
)

# ---------------------------------
# Predict Income Cluster
# ---------------------------------
# Find nearest income value for the given age
mean_income = df.groupby('Age')['Income($)'].mean()
nearest_age = mean_income.index[
    np.abs(mean_income.index - age_input).argmin()
]
estimated_income = mean_income.loc[nearest_age]

prediction = kmeans.predict([[age_input, estimated_income]])

# ---------------------------------
# Output Display
# ---------------------------------
st.subheader("📊 Prediction Result")

st.write(f"**Entered Age:** {age_input}")
st.write(f"**Estimated Income ($):** {int(estimated_income)}")
st.success(f"**Income Cluster:** {int(prediction[0])}")

# ---------------------------------
# Plot
# ---------------------------------
st.subheader("📈 Age vs Income Clustering")

fig, ax = plt.subplots()

ax.scatter(
    df['Age'],
    df['Income($)'],
    c=df['Cluster'],
    cmap='viridis',
    label='Existing Data'
)

ax.scatter(
    age_input,
    estimated_income,
    color='red',
    s=200,
    marker='X',
    label='Your Input'
)

ax.set_xlabel("Age")
ax.set_ylabel("Income ($)")
ax.set_title("K-Means Income Clustering")
ax.legend()

st.pyplot(fig)

# ---------------------------------
# Show Dataset
# ---------------------------------
with st.expander("📄 View Dataset"):
    st.dataframe(df)
