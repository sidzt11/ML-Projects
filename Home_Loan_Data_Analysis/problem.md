# Deep Learning with Keras and TensorFlow  
## Course-End Project  
## Home Loan Data Analysis  

---

## Problem Statement  

For a safe and secure lending experience, it's important to analyze historical loan data.  
In this project, the objective is to build a deep learning model that predicts the probability of loan default for future applicants using past data.

The dataset provided is highly imbalanced and contains multiple features, making this problem more challenging.

---

## Objective  

Create a deep learning model that predicts whether an applicant will repay a loan using historical data.

---

## Domain  

Finance

---

## Dataset  

The dataset is provided in CSV format and contains historical loan application data, including a target column named `TARGET`, where:

- `0` → Loan repaid  
- `1` → Loan default  

---

## Analysis To Be Done  

Perform data preprocessing and build a deep learning prediction model.

---

## Steps To Be Done  

1. Load the dataset provided.
2. Check for null values in the dataset.
3. Print the percentage of defaulters and non-defaulters in the `TARGET` column.
4. Balance the dataset if it is imbalanced.
5. Plot the balanced or imbalanced data distribution.
6. Encode categorical columns as required for modeling.
7. Build a deep learning model using Keras and TensorFlow.
8. Calculate Sensitivity (Recall for defaulters).
9. Calculate the Area Under the ROC Curve (AUC).

---

## Expected Outcome  

A trained deep learning model capable of predicting loan default risk, along with evaluation metrics including:

- Accuracy  
- Sensitivity (Recall)  
- ROC-AUC Score  