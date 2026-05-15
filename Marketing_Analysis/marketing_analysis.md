
# Marketing Campaign Analysis

## Project Overview
This project analyzes a marketing campaign dataset to understand customer behavior and the effectiveness of different marketing channels. The goal is to provide data-driven insights to the marketing team to help them tailor their strategies for better customer acquisition and retention.

## Data Science Pipeline

### 1. Data Loading and Initial Inspection
- The dataset is loaded using pandas.
- An initial inspection is performed to understand the data types and identify missing values.
- The `Dt_Customer` column is converted to a datetime format for time-based analysis.

### 2. Data Cleaning and Preprocessing
- **Missing Value Imputation**: The `Income` column has missing values. These are imputed by taking the mean income of customers with similar `Education` and `Marital_Status`.
- **Data Cleaning**: The `Education` and `Marital_Status` columns are cleaned to standardize the categories. For example, 'Graduation' is mapped to 'Graduate', and various single statuses are grouped into 'Single'.

### 3. Feature Engineering
- **Total Children**: A new feature `Children` is created by summing `Kidhome` and `Teenhome`.
- **Age**: Customer `Age` is calculated based on their `Year_Birth`.
- **Total Spending**: `Total_Spending` is calculated by summing up all expenditure columns (`MntWines`, `MntFruits`, etc.).
- **Total Purchases**: `Total_Purchases` is the sum of purchases made through all channels.

### 4. Outlier Treatment
- Outliers in the `Income` and `Age` columns are identified using the Interquartile Range (IQR) method.
- Data points lying outside 1.5 times the IQR are removed to prevent them from skewing the analysis.

### 5. Encoding Categorical Variables
- **Ordinal Encoding**: The `Education` column is ordinally encoded to represent the hierarchy in educational qualifications.
- **One-Hot Encoding**: The `Marital_Status` column is one-hot encoded to convert it into a numerical format suitable for machine learning models.

### 6. Exploratory Data Analysis (EDA)
- A **correlation heatmap** is generated to visualize the relationships between different variables.
- **Box plots and histograms** are used to understand the distribution of key variables like `Age`, `Income`, and `Total_Spending`.

### 7. Hypothesis Testing
Several hypotheses are tested to validate assumptions about customer behavior:
- **Age and Shopping Preference**: Testing if older customers prefer in-store shopping.
- **Children and Online Shopping**: Testing if customers with children are more inclined to shop online.
- **Channel Cannibalization**: Investigating if online and catalog sales are cannibalizing in-store sales.
- **Geographical Performance**: Comparing the total purchase volumes between the US and the rest of the world.

### 8. Visualization and Insights
- **Product Performance**: Identifying the best and worst-performing products based on total revenue.
- **Campaign Acceptance vs. Age**: Analyzing the relationship between customer age and their acceptance of the last marketing campaign.
- **Campaign Acceptance by Country**: Determining which country has the highest number of customers who accepted the last campaign.
- **Spending Patterns**: Investigating the relationship between the number of children and total spending.
- **Customer Complaints**: Analyzing the educational background of customers who have lodged complaints.

## Conclusion
The analysis provides actionable insights that can help the marketing team to:
- Target specific customer segments more effectively.
- Optimize the marketing mix across different channels.
- Improve product offerings based on performance.
- Enhance customer satisfaction by addressing complaints from specific customer groups.
