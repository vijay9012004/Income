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
st.title("💰 Income Group Finder by Age")
st.write("Enter an Age to see the average Income and Income Group.")

st.sidebar.header("User Input")
input_age = st.sidebar.slider(
    "Select Age",
    int(df['Age'].min()),
    int(df['Age'].max()),
    int(df['Age'].mean())
)

# ---------------- Logic: Find Income for Age ----------------
# 1. Find all rows with this Age
age_data = df[df['Age'] == input_age]

# 2. If no exact match, find closest Age
if age_data.empty:
    closest_age = df.iloc[(df['Age'] - input_age).abs().argsort()[:1]]  # closest Age
    age_data = closest_age
    st.info(f"No exact match for Age {input_age}, showing closest Age {int(closest_age['Age'].values[0])}")

# 3. Compute average income and income group
avg_income = age_data['Income($)'].mean()
group = income_group(avg_income)

# ---------------- Display Result ----------------
st.subheader(f"Age {input_age} corresponds to:")
col1, col2 = st.columns(2)
col1.metric("Average Income", f"${avg_income:,.2f}")
col2.metric("Income Group", group)

st.info(f"""
**Statistics for this Age:**
- Number of people: {len(age_data)}
- Average Income: ${avg_income:,.2f}
- Income Group: {group}
""")

# ---------------- Dataset Display ----------------
st.divider()
if st.checkbox("Show Dataset"):
    st.dataframe(df, use_container_width=True)
