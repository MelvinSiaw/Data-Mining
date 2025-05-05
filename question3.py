import pandas as pd

# Load dataset
df = pd.read_csv("Marital_satisfaction_Data.csv")

# Step 1: Display all column names for confirmation
print("📌 Column Names:\n", df.columns.tolist())

# Step 2: Filter respondents who are religious and financially struggling
religious_threshold = 5
struggling_values = [4, 5]  # Worse than average or much worse

filtered_df = df[(df["Religiosity"] >= religious_threshold) & (df["Material status"].isin(struggling_values))]

print(f"\n🧮 Total filtered respondents (Religious ≥ {religious_threshold} and Financially struggling): {len(filtered_df)}")

# Step 3: Use Unnamed:16 (Q15 - Cuddling) and Unnamed:17 (Q16 - Respect)
# Rename for clarity
bonding_df = filtered_df[["Unnamed: 16", "Unnamed: 17"]].copy()
bonding_df.columns = ["Q15_Cuddling", "Q16_Respect"]

# Drop any rows with missing values
bonding_df = bonding_df.dropna()

# Step 4: Convert to numeric in case of string types
bonding_df = bonding_df.astype(float)

# Step 5: Compute average bonding scores
avg_scores = bonding_df.mean()

print("\n❤️ Average Emotional Bonding Scores:")
print(avg_scores)

# Step 6: Interpret results
if (avg_scores > 0).all():  # > 0 on a -2 to +2 scale indicates strong bonding
    print("\n✅ Emotional bonding is still strong among religious and financially struggling respondents.")
else:
    print("\n⚠️ Emotional bonding appears weak or mixed.")
