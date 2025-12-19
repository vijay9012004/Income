import streamlit as st
import pandas as pd
import numpy as np
import os
import kagglehub

# ---------------- Page Config ----------------
st.set_page_config(page_title="Income Group Finder", page_icon="💰")

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("duajanmuhammed/kmean-data")
    csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
    return pd.read_csv(os.path.join(path, csv_file))

df = load_data()

# ---------------- Create Income Groups ----------------
# Rule-based grouping (NO ML)
low_thresh = df['Income($)'].quantile(0.33)
high_thresh = df['Income($)'].quantile(0.66)

def income_group(income):
    if income <= low_thresh:
        return "Low Income Group"
    elif income <= high_thresh:
        return "Middle Income Group"
    else:
        return "High Income Group"

df["Income Group"] = df["Income($)"].apply(income_group)

# ---------------- UI ----------------
st.title("💰 Income Group Finder (Without ML)")
st.write("Enter age and income to determine income category.")

st.sidebar.header("User Input")
input_age = st.sidebar.slider(
    "Select Age",
    int(df['Age'].min()),
    int(df['Age'].max()),
    int(df['Age'].mean())
)

input_income = st.sidebar.number_input(
    "Enter Income ($)",
    value=int(df['Income($)'].median()),
    step=1000
)

# ---------------- Prediction ----------------
if st.sidebar.button("Find Income Group"):
    group = income_group(input_income)

    st.subheader(f"Result: {group}")

    col1, col2 = st.columns(2)
    col1.metric("Age", input_age)
    col2.metric("Income", f"${input_income:,}")

    group_data = df[df["Income Group"] == group]

    st.info(f"""
    **Group Statistics**
    - Total people: {len(group_data)}
    - Average Age: {group_data['Age'].mean():.1f}
    - Average Income: ${group_data['Income($)'].mean():,.2f}
    """)
else:
    st.info("Enter values in the sidebar and click **Find Income Group**.")

# ---------------- Dataset Display ----------------
st.divider()
if st.checkbox("Show Dataset"):
    st.dataframe(df, use_container_width=True)
