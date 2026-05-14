import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv('Sales Analysis/1736837073_ausapparalsales4thqrt2020/AusApparalSales4thQrt2020.csv')

# Data wrangling
print(df.isna().sum())
print(df.dtypes)
print(df.head())
df_clean = df.dropna()
scaler = MinMaxScaler()
cols_to_normalize = ['Sales', 'Units']
df_clean[cols_to_normalize] = scaler.fit_transform(df_clean[cols_to_normalize])

# Descriptive statistics
desc_stats = df_clean[cols_to_normalize].describe()
print(desc_stats)
print(df_clean[cols_to_normalize].mode())

# Top/bottom groups by sales
state_sales = df_clean.groupby('State')['Sales'].sum().sort_values(ascending=False)
print(state_sales)
top_state = state_sales.idxmax()
bottom_state = state_sales.idxmin()
group_sales = df_clean.groupby('Group')['Sales'].sum().sort_values(ascending=False)
print(group_sales)
top_group = group_sales.idxmax()
bottom_group = group_sales.idxmin()

# Time-based aggregations
df_clean['Date'] = pd.to_datetime(df_clean['Date'])
df_clean['Week'] = df_clean['Date'].dt.isocalendar().week
df_clean['Month'] = df_clean['Date'].dt.month
df_clean['Quarter'] = df_clean['Date'].dt.quarter
weekly_sales = df_clean.groupby('Week')['Sales'].sum()
monthly_sales = df_clean.groupby('Month')['Sales'].sum()
quarterly_sales = df_clean.groupby('Quarter')['Sales'].sum()
print(weekly_sales)
print(monthly_sales)
print(quarterly_sales)

# Visualizations
sns.set(style="whitegrid")
plt.figure(figsize=(10,6))
sns.barplot(data=df_clean, x='State', y='Sales', hue='Group', estimator=sum)
plt.title('State-wise Sales by Group')
plt.show()
plt.figure(figsize=(10,6))
sns.barplot(data=df_clean, x='Group', y='Sales', hue='State', estimator=sum)
plt.title('Group-wise Sales by State')
plt.show()
if 'Time' in df_clean.columns:
    df_clean['Hour'] = pd.to_datetime(df_clean['Time'], errors='coerce').dt.hour
    plt.figure(figsize=(10,6))
    sns.lineplot(data=df_clean, x='Hour', y='Sales', estimator='sum')
    plt.title('Sales by Hour of Day')
    plt.show()
plt.figure(figsize=(8,5))
sns.boxplot(data=df_clean[cols_to_normalize])
plt.title('Boxplot of Normalized Sales and Units')
plt.show()
plt.figure(figsize=(8,5))
sns.histplot(df_clean['Sales'], kde=True)
plt.title('Distribution of Normalized Sales')
plt.show()
