import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

# Load the dataset
data = pd.read_csv('https://www.kaggle.com/liujiaqi/hr-comma-sepcsv/download')

# 1. Data Quality Check
print(data.isnull().sum())

# 2. EDA
# 2.1. Correlation Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

# 2.2. Distribution Plots
sns.distplot(data['satisfaction_level'])
plt.show()
sns.distplot(data['last_evaluation'])
plt.show()
sns.distplot(data['average_montly_hours'])
plt.show()

# 2.3. Project Count Bar Plot
sns.countplot(x='number_project', hue='left', data=data)
plt.show()

# 3. Clustering
left_employees = data[data['left'] == 1][['satisfaction_level', 'last_evaluation']]
kmeans = KMeans(n_clusters=3, random_state=42)
left_employees['cluster'] = kmeans.fit_predict(left_employees)

# 4. Class Imbalance Handling
# 4.1. Pre-processing
categorical_cols = ['Department', 'salary']
numerical_cols = data.columns.drop(categorical_cols + ['left'])
data_dummies = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
X = data_dummies.drop('left', axis=1)
y = data_dummies['left']

# 4.2. Stratified Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

# 4.3. SMOTE
smote = SMOTE(random_state=123)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 5. Model Training and Evaluation
# 5.1. Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr_scores = cross_val_score(lr, X_train_smote, y_train_smote, cv=5)
print("Logistic Regression CV Scores:", lr_scores)
lr.fit(X_train_smote, y_train_smote)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, lr.predict(X_test)))

# 5.2. Random Forest
rf = RandomForestClassifier(random_state=123)
rf_scores = cross_val_score(rf, X_train_smote, y_train_smote, cv=5)
print("Random Forest CV Scores:", rf_scores)
rf.fit(X_train_smote, y_train_smote)
print("Random Forest Classification Report:")
print(classification_report(y_test, rf.predict(X_test)))

# 5.3. Gradient Boosting
gb = GradientBoostingClassifier(random_state=123)
gb_scores = cross_val_score(gb, X_train_smote, y_train_smote, cv=5)
print("Gradient Boosting CV Scores:", gb_scores)
gb.fit(X_train_smote, y_train_smote)
print("Gradient Boosting Classification Report:")
print(classification_report(y_test, gb.predict(X_test)))

# 6. Model Comparison
# 6.1. ROC/AUC
models = {'Logistic Regression': lr, 'Random Forest': rf, 'Gradient Boosting': gb}
for name, model in models.items():
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
plt.legend()
plt.show()

# 6.2. Confusion Matrix
for name, model in models.items():
    print(f"Confusion Matrix for {name}:")
    print(confusion_matrix(y_test, model.predict(X_test)))

# 7. Retention Strategies
# 7.1. Predict Probability
y_pred_proba_test = rf.predict_proba(X_test)[:, 1]
results = pd.DataFrame({'probability': y_pred_proba_test})

# 7.2. Categorize Employees
def categorize(score):
    if score < 0.2:
        return 'Safe Zone (Green)'
    elif score < 0.6:
        return 'Low-Risk Zone (Yellow)'
    elif score < 0.9:
        return 'Medium-Risk Zone (Orange)'
    else:
        return 'High-Risk Zone (Red)'
results['zone'] = results['probability'].apply(categorize)
print(results['zone'].value_counts())
