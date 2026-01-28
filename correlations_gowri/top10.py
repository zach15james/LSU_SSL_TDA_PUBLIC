
import pandas as pd
import matplotlib.pyplot as plt

# Load correlation weights
df = pd.read_csv("correlation_weights_ALM.csv")

# Sort by absolute correlation and get top 10
top_corr = df.reindex(df['Corr'].abs().sort_values(ascending=False).index)
top_10 = top_corr.head(10)

# Save to CSV
top_10.to_csv("top10_correlated_features_ALM.csv", index=False)

print("Top 10 features saved to 'top10_correlated_features_ALM.csv'")
print(top_10)


# Plot horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(top_10['Feature'], top_10['Corr'], color=['green' if c > 0 else 'red' for c in top_10['Corr']])
plt.xlabel("Correlation with ALM")
plt.title("Top 10 Features Most Correlated with ALM")
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.gca().invert_yaxis()  # Highest correlation at top
plt.grid(True, axis='x', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("top10_correlated_features_ALM.png")
plt.show()
