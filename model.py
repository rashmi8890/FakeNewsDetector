import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

folder_path = r"C:\Users\dell\Downloads\archive\News _dataset"

df_fake = pd.read_csv(os.path.join(folder_path,"Fake.csv"))
df_real = pd.read_csv(os.path.join(folder_path,"True.csv"))

df_fake["label"] = 0
df_real["label"] = 1

df = pd.concat([df_fake,df_real],ignore_index=True)
df = df.sample(frac=1,random_state=42).reset_index(drop=True)

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

vectorizer=TfidfVectorizer(stop_words='english',max_df=0.7)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model=LogisticRegression()

model.fit(X_train_vec,y_train)

y_pred = model.predict(X_test_vec)

print(classification_report(y_pred,y_test))

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")