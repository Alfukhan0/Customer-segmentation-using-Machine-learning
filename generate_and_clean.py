import pandas as pd
import numpy as np
import random

# 1. Generate the Extended Dataset with messy data
np.random.seed(42)
random.seed(42)

n_samples = 300

# Base parameters similar to original
customer_ids = range(1, n_samples + 1)
genders = random.choices(['Male', 'Female', 'male', 'FEMALE', 'M', 'F', None], weights=[45, 45, 2, 2, 2, 2, 2], k=n_samples)
ages = np.random.normal(38, 15, n_samples)
annual_incomes = np.random.randint(10000, 75001, n_samples).astype(float)  # 4-5 digit integers, no decimals
spending_scores = np.random.normal(50, 25, n_samples)

# New parameters
professions = ['Healthcare', 'Engineer', 'Lawyer', 'Entertainment', 'Artist', 'Executive', 'Doctor', 'Homemaker', 'Marketing']
profession_choices = random.choices(professions + [None], weights=[1]*9 + [2], k=n_samples)
work_experiences = np.random.normal(5, 4, n_samples)
family_sizes = np.random.poisson(3, n_samples)
purchase_frequencies = np.random.poisson(15, n_samples)
memberships = random.choices(['Standard', 'Silver', 'Gold', 'Platinum'], k=n_samples)

# Induce some noise and dirty data
ages[10:15] = -5 # Negative ages
ages[20:25] = 200 # Unrealistic ages
ages[30:35] = np.nan # Missing ages

annual_incomes[5:10] = -5000  # Negative incomes (dirty data)
annual_incomes[60:65] = np.nan  # Missing incomes (dirty data)
spending_scores = np.clip(spending_scores, -20, 150) # Scores outside 1-100 range

work_experiences[work_experiences < 0] = 0
work_experiences[40:50] = np.nan

df = pd.DataFrame({
    'CustomerID': customer_ids,
    'Gender': genders,
    'Age': ages,
    'Annual Income ($)': annual_incomes,
    'Spending Score (1-100)': spending_scores,
    'Profession': profession_choices,
    'Work Experience': work_experiences,
    'Family Size': family_sizes,
    'Purchase Frequency': purchase_frequencies,
    'Membership': memberships
})

# Save the dirty dataset
dirty_file = 'Extended_Mall_Customers_Dirty.csv'
df.to_csv(dirty_file, index=False)
print(f"Generated dirty dataset: {dirty_file}")

# 2. Clean the Dataset using Python
print("Starting data cleaning...")
df_clean = df.copy()

# A. Standardize Gender
gender_map = {
    'male': 'Male', 'MALE': 'Male', 'M': 'Male', 
    'FEMALE': 'Female', 'female': 'Female', 'F': 'Female'
}
df_clean['Gender'] = df_clean['Gender'].replace(gender_map)
# Fill missing Gender with mode
df_clean['Gender'] = df_clean['Gender'].fillna(df_clean['Gender'].mode()[0])

# B. Clean Age
# Replace negative and unrealistic ages with NaN, then impute with median
df_clean.loc[(df_clean['Age'] < 0) | (df_clean['Age'] > 100), 'Age'] = np.nan
df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median()).astype(int)

# C. Clean Annual Income
# Replace negative incomes with NaN, fill with median, convert to int (no decimals)
df_clean.loc[df_clean['Annual Income ($)'] < 0, 'Annual Income ($)'] = np.nan
df_clean['Annual Income ($)'] = df_clean['Annual Income ($)'].fillna(df_clean['Annual Income ($)'].median())
df_clean['Annual Income ($)'] = df_clean['Annual Income ($)'].astype(int)  # No decimal values

# D. Clean Spending Score
# Clip between 1 and 100
df_clean['Spending Score (1-100)'] = df_clean['Spending Score (1-100)'].clip(1, 100).astype(int)

# E. Clean Profession
# Fill missing with 'Unknown'
df_clean['Profession'] = df_clean['Profession'].fillna('Unknown')

# F. Clean Work Experience
df_clean['Work Experience'] = df_clean['Work Experience'].fillna(df_clean['Work Experience'].median()).astype(int)

# Ensure data types
df_clean['Family Size'] = df_clean['Family Size'].astype(int)
df_clean['Purchase Frequency'] = df_clean['Purchase Frequency'].astype(int)

# Save the cleaned dataset
clean_file = 'Extended_Mall_Customers_Clean.csv'
df_clean.to_csv(clean_file, index=False)
print(f"Generated clean dataset: {clean_file}")
