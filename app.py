import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(layout="wide")

st.title("🧠 EEG Signal Based Abnormality Detection Dashboard")

# Upload File
uploaded_file = st.file_uploader("Upload EEG CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------
    # Create Status Column
    # ---------------------------
    df["Status"] = df["Eye Detection"].apply(
        lambda x: "Normal" if x == 0 else "Abnormal"
    )

    # ---------------------------
    # KPI Section
    # ---------------------------
    total = len(df)
    normal_percent = (df["Status"] == "Normal").mean() * 100
    abnormal_percent = (df["Status"] == "Abnormal").mean() * 100

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", total)
    col2.metric("Normal %", f"{normal_percent:.2f}%")
    col3.metric("Abnormal %", f"{abnormal_percent:.2f}%")

    # ---------------------------
    # Pie Chart
    # ---------------------------
    st.subheader("Status Distribution")

    fig1 = px.pie(df, names="Status", title="Normal vs Abnormal Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    # ---------------------------
    # Pivot EEG Channels
    # ---------------------------
    eeg_columns = df.columns[:-2]  # exclude Eye Detection & Status
    eeg_df = df.melt(
        id_vars=["Status"],
        value_vars=eeg_columns,
        var_name="Channel",
        value_name="Signal_Value"
    )

    # ---------------------------
    # Bar Chart
    # ---------------------------
    st.subheader("Average EEG Signal per Channel")

    fig2 = px.bar(
        eeg_df,
        x="Channel",
        y="Signal_Value",
        color="Status",
        barmode="group",
        title="Channel-wise Signal Comparison"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ---------------------------
    # Heatmap
    # ---------------------------
    st.subheader("Heatmap: Channel vs Status")

    heatmap_data = eeg_df.groupby(["Status", "Channel"])["Signal_Value"].mean().unstack()

    fig3, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=False)
    st.pyplot(fig3)

    # ---------------------------
    # Butterfly Chart
    # ---------------------------
    st.subheader("Butterfly Chart (Normal vs Abnormal)")

    butterfly = eeg_df.groupby(["Channel", "Status"])["Signal_Value"].mean().unstack()

    butterfly["Abnormal"] = -butterfly["Abnormal"]

    fig4, ax = plt.subplots(figsize=(8, 6))
    ax.barh(butterfly.index, butterfly["Normal"], color="blue")
    ax.barh(butterfly.index, butterfly["Abnormal"], color="red")
    ax.set_title("Signal Difference (Normal vs Abnormal)")
    st.pyplot(fig4)