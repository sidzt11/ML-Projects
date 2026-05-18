# ============================================================
# Home Loan Default Prediction using Deep Learning
# Compatible with Google Colab & Jupyter Notebook
# ============================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.utils import resample

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#  Load Dataset
df = pd.read_csv("loan_data.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:\n", df.head())

#  Check for Null Values
print("\nNull Values:\n", df.isnull().sum())

# Fill numerical nulls with median
for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical nulls with mode
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nNull Values After Handling:\n", df.isnull().sum().sum())

#  Check Target Distribution
target_counts = df['TARGET'].value_counts()
print("\nTarget Distribution:\n", target_counts)

percentage = df['TARGET'].value_counts(normalize=True) * 100
print("\nPercentage Distribution:\n", percentage)

#  Plot Imbalance
plt.figure(figsize=(6,4))
sns.countplot(x='TARGET', data=df)
plt.title("Target Distribution (Before Balancing)")
plt.show()

#  Balance Dataset (Upsampling Minority Class)
df_majority = df[df.TARGET == 0]
df_minority = df[df.TARGET == 1]

if len(df_minority) < len(df_majority):
    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
else:
    df_balanced = df.copy()

print("\nBalanced Target Distribution:\n", df_balanced['TARGET'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x='TARGET', data=df_balanced)
plt.title("Target Distribution (After Balancing)")
plt.show()

#  Encode Categorical Columns
label_encoders = {}
for col in df_balanced.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_balanced[col] = le.fit_transform(df_balanced[col])
    label_encoders[col] = le

#  Split Features and Target
X = df_balanced.drop("TARGET", axis=1)
y = df_balanced["TARGET"]

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

#  Build Deep Learning Model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(32, activation='relu'),

    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

#  Train Model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    verbose=1
)

#  Evaluate Model
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

#  Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

#  Sensitivity (Recall for Default = 1)
TP = cm[1,1]
FN = cm[1,0]
sensitivity = TP / (TP + FN)
print("\nSensitivity (Recall for Default Class):", sensitivity)

#  ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred_prob)
print("\nROC-AUC Score:", roc_auc)

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

print("\nModel Training and Evaluation Completed Successfully!")