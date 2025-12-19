import streamlit as st
import pandas as pd
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(page_title="Income Group Finder", page_icon="💰")

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    return pd.read_csv("income_data.csv")

df = load_data()

# ---------------- Create Income Groups (Rule-based) ----------------
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
st.title("💰 Income Group Finder")
st.write("Enter **Age** and **Income** to identify the income category.")

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

st.caption(
    f"Income range in dataset: "
    f"${int(df['Income($)'].min()):,} – ${int(df['Income($)'].max()):,}"
)

# ---------------- Result ----------------
if st.sidebar.button("Find Income Group"):
    group = income_group(input_income)

    st.subheader(f"Result: {group}")

    col1, col2 =
