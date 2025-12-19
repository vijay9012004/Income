# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import kagglehub

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Income K-Means Clustering",
    layout="centered"
)

st.title("💰 Income Analysis using K-Means Clustering")
st.write("Clustering people based on **Age** and **Income ($)**")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("duajanmuhammed/kmean-data")
    files_in_dir = os.listdir(path)
    csv_file = [f for f in files_in_dir if f.endswith('.csv')][0]
    full_csv_path = os.path.join(path, csv_file)
    return pd.read_csv(full_csv_path)

df = load_data()

# -------------------------------
# Display Dataset
# -------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df)

# -------------------------------
# Select Features
# -------------------------------
x = df[['Age', 'Income($)']]

# -------------------------------
# K-Means Model
# -------------------------------
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
df['Cluster'] = kmeans.fit_predict(x)

# -------------------------------
# Display Clustered Data
# -------------------------------
st.subheader("📊 Clustered Income Data")
st.dataframe(df)

# -------------------------------
# Visualization
# -------------------------------
st.subheader("📈 Age vs Income Clustering")

fig, ax = plt.subplots()

scatter = ax.scatter(
    df['Age'],
    df['Income($)'],
    c=df['Cluster'],
    cmap='viridis'
)

ax.set_xlabel("Age")
ax.set_ylabel("Income ($)")
ax.set_title("K-Means Clustering (k = 3)")

st.pyplot(fig)

# -------------------------------
# Cluster Centers
# -------------------------------
st.subheader("🎯 Cluster Centers (Income Groups)")

centers = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=["Age", "Income($)"]
)

st.dataframe(centers)

st.success("Income clustering completed successfully!")
