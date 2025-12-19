import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import kagglehub
import os

st.set_page_config(page_title="Income KMeans Clustering", layout="centered")

st.title("💰 Income Clustering using K-Means")
st.write("K-Means clustering on Age vs Income dataset")

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("duajanmuhammed/kmean-data")
    files_in_dir = os.listdir(path)
    csv_file = [f for f in files_in_dir if f.endswith('.csv')][0]
    full_csv_path = os.path.join(path, csv_file)
    return pd.read_csv(full_csv_path)

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("Model Parameters")
k = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=6, value=3)

# ---------------------------
# Model Training
# ---------------------------
x = df[['Age', 'Income($)']]

kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
labels = kmeans.fit_predict(x)

df['Cluster'] = labels

# ---------------------------
# Visualization
# ---------------------------
st.subheader("Cluster Visualization")

fig, ax = plt.subplots()
scatter = ax.scatter(
    df['Age'],
    df['Income($)'],
    c=df['Cluster'],
    cmap='viridis'
)

ax.set_xlabel("Age")
ax.set_ylabel("Income ($)")
ax.set_title(f"K-Means Clustering (k={k})")

st.pyplot(fig)

# ---------------------------
# Cluster Centers
# ---------------------------
st.subheader("Cluster Centers")
centers = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=['Age', 'Income($)']
)
st.dataframe(centers)

st.success("Clustering completed successfully!")
