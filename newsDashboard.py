# import streamlit as st
# import pandas as pd
# import os
# import plotly.express as px
# import matplotlib.pyplot as plt
# import seaborn as sns
# from wordcloud import WordCloud
# import joblib

# model = joblib.load("model/fake_news_model.pkl")
# vectorizer = joblib.load("model/vectorizer.pkl")

# folder_path = r"C:\Users\dell\Downloads\archive\News _dataset"
# fake_df =  pd.read_csv(os.path.join(folder_path,"Fake.csv"))
# real_df = pd.read_csv(os.path.join(folder_path,"True.csv"))
# fake_df['label'] = 0
# real_df['label'] = 1
# df = pd.concat([fake_df, real_df], ignore_index=True)
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# # ----- Streamlit UI -----
# st.title("📰 Fake News Detection Dashboard")

# tab1, tab2 = st.tabs(["📊 Data Visualization", "🔍 Predict News"])

# with tab1:
#     st.subheader("Label Distribution")

#     label_counts = df['label'].value_counts().rename({0: "Fake", 1: "Real"})
#     fig_label = px.bar(
#         x=label_counts.index,
#         y=label_counts.values,
#         labels={'x': 'News Type', 'y': 'Count'},
#         title='Distribution of Fake and Real News',
#         color=label_counts.index,
#         color_discrete_map={'Fake': 'red', 'Real': 'green'}
#     )
#     st.plotly_chart(fig_label)

#     st.subheader("Article Length Distribution")

#     df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
#     df['label_name'] = df['label'].map({0: 'Fake', 1: 'Real'})

#     fig_length = px.histogram(
#         df,
#         x="text_length",
#         color="label_name",
#         nbins=50,
#         title="Distribution of News Article Lengths",
#         labels={"text_length": "Number of Words"},
#         color_discrete_map={'Fake': 'red', 'Real': 'green'}
#     )
#     st.plotly_chart(fig_length)


#     st.subheader("WordClouds")

#     fake_text = " ".join(fake_df['text'].dropna().astype(str))
#     fake_wc = WordCloud(width=800, height=400, background_color='white').generate(fake_text)
#     st.markdown("**Fake News WordCloud**")
#     st.image(fake_wc.to_array(), use_container_width=True)

#     real_text = " ".join(real_df['text'].dropna().astype(str))
#     real_wc = WordCloud(width=800, height=400, background_color='white').generate(real_text)
#     st.markdown("**Real News WordCloud**")
#     st.image(real_wc.to_array(), use_container_width=True)

# with tab2:
#     st.subheader("Enter News Text to Predict")
#     user_input = st.text_area("Paste or type a news article here:", height=200)

#     if st.button("Predict"):
#         if user_input.strip() == "":
#             st.warning("Please enter some text.")
#         else:
#             text_vec = vectorizer.transform([user_input])
#             prediction = model.predict(text_vec)[0]
#             label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
#             st.success(f"Prediction: {label}")


import streamlit as st
import pandas as pd
import os
import plotly.express as px
from wordcloud import WordCloud
import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np

# ---------- Load Model and Data ----------
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

folder_path = r"C:\Users\dell\Downloads\archive\News _dataset"
fake_df = pd.read_csv(os.path.join(folder_path, "Fake.csv"))
real_df = pd.read_csv(os.path.join(folder_path, "True.csv"))

fake_df['label'] = 0
real_df['label'] = 1

df = pd.concat([fake_df, real_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df['label_name'] = df['label'].map({0: 'Fake', 1: 'Real'})
df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detection Dashboard")

tab1, tab2 = st.tabs(["📊 Data Visualization", "🔍 Predict News"])

# -----------------------
# 📊 Tab 1: Visualizations
# -----------------------
with tab1:
    st.subheader("📌 Dataset Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Articles", len(df))
    col2.metric("Fake News", df['label'].value_counts()[0])
    col3.metric("Real News", df['label'].value_counts()[1])

    st.markdown("---")
    st.subheader("📊 Label Distribution")

    label_counts = df['label'].value_counts().rename({0: "Fake", 1: "Real"})
    fig_label = px.bar(
        x=label_counts.index,
        y=label_counts.values,
        labels={'x': 'News Type', 'y': 'Count'},
        title='Fake vs Real News Count',
        color=label_counts.index,
        color_discrete_map={'Fake': 'red', 'Real': 'green'}
    )
    st.plotly_chart(fig_label, use_container_width=True)

    st.subheader("📏 Article Length Distribution")

    fig_length = px.histogram(
        df,
        x="text_length",
        color="label_name",
        nbins=50,
        title="Distribution of Article Word Counts",
        labels={"text_length": "Number of Words"},
        color_discrete_map={'Fake': 'red', 'Real': 'green'}
    )
    st.plotly_chart(fig_length, use_container_width=True)

    st.subheader("🎨 Word Clouds")
    wc_col1, wc_col2 = st.columns(2)

    with wc_col1:
        
        #if st.checkbox("Show Fake News WordCloud"):
        fake_text = " ".join(fake_df['text'].dropna().astype(str))
        fake_wc = WordCloud(width=800, height=400, background_color='white').generate(fake_text)
        st.image(fake_wc.to_array(), caption="Fake News", use_container_width=True)

    with wc_col2:
        #if st.checkbox("Show Real News WordCloud"):
        real_text = " ".join(real_df['text'].dropna().astype(str))
        real_wc = WordCloud(width=800, height=400, background_color='white').generate(real_text)
        st.image(real_wc.to_array(), caption="Real News", use_container_width=True)

# -----------------------
# 🔍 Tab 2: Prediction
# -----------------------
with tab2:
    st.subheader("✍️ Enter News Text")
    user_input = st.text_area("Paste a news article or write your own text here:", height=200)

    col1, col2 = st.columns([1, 2])
    with col1:
        show_tokens = st.checkbox("Preview processed tokens")

    with col2:
        btn = st.button("🚀 Predict", use_container_width=True)

    if btn:
        if not user_input.strip():
            st.warning("Please enter a news article.")
        else:
            # Preprocessing and prediction
            transformed = vectorizer.transform([user_input])
            prediction = model.predict(transformed)[0]
            probas = model.predict_proba(transformed)[0]
            confidence = round(np.max(probas) * 100, 2)
            label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"

            st.markdown(f"### Prediction: {label}")
            st.progress(confidence / 100)
            st.write(f"Confidence Score: **{confidence}%**")

            if show_tokens:
                tokens = vectorizer.inverse_transform(transformed)[0]
                st.info(f"Processed tokens used for prediction: `{', '.join(tokens[:30])}...`")

