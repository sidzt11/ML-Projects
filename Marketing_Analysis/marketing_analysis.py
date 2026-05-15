
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Load the dataset
try:
    df = pd.read_csv('C:/Projects/ML/Sales Analysis/1736837073_ausapparalsales4thqrt2020/AusApparalSales4thQrt2020.csv', encoding='utf-8')
except FileNotFoundError:
    print("Error: Data file not found. Please check the file path.")
    exit()


# 1. Data Inspection
print("Initial Data Info:")
df.info()
print("\nChecking for missing values:")
print(df.isnull().sum())

# Convert 'Dt_Customer' to datetime
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])

# 2. Missing Value Imputation for 'Income'
# Clean 'Education' and 'Marital_Status'
df['Education'] = df['Education'].str.replace('Graduation', 'Graduate')
df['Marital_Status'] = df['Marital_Status'].replace(['Together', 'Married'], 'Partner')
df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'Absurd', 'YOLO'], 'Single')

# Impute 'Income' based on the mean income of similar 'Education' and 'Marital_Status' groups
df['Income'] = df.groupby(['Education', 'Marital_Status'])['Income'].transform(lambda x: x.fillna(x.mean()))

# 3. Feature Engineering
# Total children
df['Children'] = df['Kidhome'] + df['Teenhome']
# Age
df['Age'] = 2023 - df['Year_Birth']
# Total spending
spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df['Total_Spending'] = df[spending_cols].sum(axis=1)
# Total purchases
purchase_cols = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
df['Total_Purchases'] = df[purchase_cols].sum(axis=1)

# 4. Outlier Treatment
# Using the IQR method for 'Income' and 'Age'
for col in ['Income', 'Age']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# 5. Encoding Categorical Variables
# Ordinal encoding for 'Education'
education_map = {'Basic': 0, '2n Cycle': 1, 'Graduate': 2, 'Master': 3, 'PhD': 4}
df['Education_Encoded'] = df['Education'].map(education_map)
# One-hot encoding for 'Marital_Status'
df = pd.get_dummies(df, columns=['Marital_Status'], prefix='Marital')

# 6. Correlation Heatmap
plt.figure(figsize=(18, 12))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Variables')
plt.show()

# 7. Hypothesis Testing
# a. Older vs. In-store shopping
older_group = df[df['Age'] > df['Age'].median()]
younger_group = df[df['Age'] <= df['Age'].median()]
t_stat, p_val = ttest_ind(older_group['NumStorePurchases'], younger_group['NumStorePurchases'])
print(f"\nHypothesis a: p-value = {p_val:.3f}. {'Reject' if p_val < 0.05 else 'Fail to reject'} the null hypothesis.")

# b. Customers with children vs. Online shopping
with_children = df[df['Children'] > 0]
no_children = df[df['Children'] == 0]
t_stat, p_val = ttest_ind(with_children['NumWebPurchases'], no_children['NumWebPurchases'])
print(f"Hypothesis b: p-value = {p_val:.3f}. {'Reject' if p_val < 0.05 else 'Fail to reject'} the null hypothesis.")

# c. Cannibalization of store sales
t_stat, p_val = ttest_ind(df['NumStorePurchases'], df['NumWebPurchases'] + df['NumCatalogPurchases'])
print(f"Hypothesis c: p-value = {p_val:.3f}. {'Reject' if p_val < 0.05 else 'Fail to reject'} the null hypothesis.")

# d. US vs. Rest of the World in total purchases
# Assuming 'Country' column exists. If not, this will fail.
if 'Country' in df.columns:
    us_purchases = df[df['Country'] == 'USA']['Total_Purchases']
    row_purchases = df[df['Country'] != 'USA']['Total_Purchases']
    t_stat, p_val = ttest_ind(us_purchases, row_purchases)
    print(f"Hypothesis d: p-value = {p_val:.3f}. {'Reject' if p_val < 0.05 else 'Fail to reject'} the null hypothesis.")
else:
    print("Hypothesis d: 'Country' column not found. Skipping this test.")


# 8. Visualization
# a. Top/Bottom performing products
product_performance = df[spending_cols].sum().sort_values(ascending=False)
print("\nProduct Performance (Total Spending):")
print(product_performance)

# b. Age vs. Last campaign acceptance
sns.boxplot(x='AcceptedCmp5', y='Age', data=df)
plt.title('Age Distribution by Last Campaign Acceptance')
plt.show()

# c. Country with highest campaign acceptance
if 'Country' in df.columns:
    country_acceptance = df.groupby('Country')['AcceptedCmp5'].sum().sort_values(ascending=False)
    print("\nCountry with highest number of last campaign acceptances:")
    print(country_acceptance)
else:
    print("\n'Country' column not found. Cannot analyze campaign acceptance by country.")


# d. Children at home vs. Total expenditure
sns.boxplot(x='Children', y='Total_Spending', data=df)
plt.title('Total Spending by Number of Children')
plt.show()

# e. Education of customers with complaints
if 'Complain' in df.columns:
    complaint_edu = df[df['Complain'] == 1]['Education'].value_counts()
    print("\nEducational background of customers who complained:")
    print(complaint_edu)
else:
    print("\n'Complain' column not found. Cannot analyze complaints by education.")

print("\nMarketing analysis complete.")
