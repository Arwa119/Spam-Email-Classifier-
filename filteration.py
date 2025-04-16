import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv('emails.csv')
print(df.isnull().sum())
df = df.fillna('')
X = df.iloc[:,1:-1]
y = df.iloc[:,-1]
print(f"\nFeature shape: {X.shape}")
print(f"Target variable shape: {y.shape}")

X_text = X.apply(lambda row: ''.join(row.values.astype(str)),axis=1 )
vectorizer = CountVectorizer(max_features=1000)
X_features = vectorizer.fit_transform(X_text)
print(f"Shape of X-features{X_features.shape}")
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
print(f"Performance of Linear Regression:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1 Score: {f1:.4f}")

