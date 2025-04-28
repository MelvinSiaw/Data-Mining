# Full pipeline: Data Exploration, Preprocessing, Association Rule Mining

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset
file_path = 'marital_satisfaction_data.xlsx'
data = pd.read_excel(file_path)

# --- Data Exploration ---
print("Summary Statistics:")
print(data.describe())

# Visualization examples
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'].dropna(), bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['Marriage_duration'].dropna(), bins=20, kde=True)
plt.title('Marriage Duration Distribution')
plt.xlabel('Years')
plt.ylabel('Count')
plt.show()

# --- Preprocessing ---
# Binning Age, Marriage Duration, Religiosity
bins_age = [0, 25, 35, 45, 60, 100]
labels_age = ['<25', '25-34', '35-44', '45-59', '60+']
data['AgeGroup'] = pd.cut(data['Age'], bins=bins_age, labels=labels_age)

bins_duration = [0, 5, 10, 20, 30, 50]
labels_duration = ['0-4', '5-9', '10-19', '20-29', '30+']
data['MarriageDurationGroup'] = pd.cut(data['Marriage_duration'], bins=bins_duration, labels=labels_duration)

# Average Satisfaction (simplified)
data['Avg_Satisfaction'] = data[[f'Satisfaction_Q{i}' for i in range(1, 21)]].mean(axis=1)
data['Satisfaction_Level'] = pd.cut(data['Avg_Satisfaction'], bins=[0, 4, 5, 6, 7], labels=['Low', 'Medium', 'High', 'Very High'])

# Keep necessary columns for mining
mining_data = data[['Sex', 'AgeGroup', 'MarriageDurationGroup', 'Num_children', 'Education',
                    'Material_status', 'Religion', 'Religiosity', 'Collectivism_Individualism_National',
                    'Satisfaction_Level']]

# One-hot encoding for association rules
transactions = pd.get_dummies(mining_data.astype(str))

# --- Association Rule Mining ---
frequent_itemsets = apriori(transactions, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

# Sort by lift and confidence
rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])

# Show top rules
print("Top Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
