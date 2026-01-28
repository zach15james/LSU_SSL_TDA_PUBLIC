

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("penn_data.csv")

# Define target variables
all_targets = ['ALM', 'BMD', 'BFP']

# Extract features (exclude target columns)
feature_df = df.drop(columns=all_targets).select_dtypes(include=[np.number])
X = feature_df.values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Process each target
for target in all_targets:
    print(f"\n Processing target: {target}")
    y = df[target].values

    # Compute correlation for each feature with the target
    correlations = np.array([
        np.corrcoef(X_scaled[:, j], y)[0, 1]
        for j in range(X_scaled.shape[1])
    ])

    # Save full correlation table
    corr_df = pd.DataFrame({
        "Feature": feature_df.columns,
        "Correlation": correlations
    }).round(4)
    corr_df.to_csv(f"correlation_only_{target}.csv", index=False)

    # Plot all feature correlations
    x = np.arange(len(feature_df.columns))
    plt.figure(figsize=(14, 6))
    plt.plot(x, correlations, label="Correlation", marker='o', color='blue', linewidth=2)
    for i, corr in enumerate(correlations):
        plt.text(i, corr + 0.02, f"{corr:.2f}", ha='center', va='bottom', fontsize=8, color='blue', rotation=90)
    plt.xticks(x, feature_df.columns, rotation=90)
    plt.ylabel("Correlation")
    plt.title(f"Feature Correlation with {target}")
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"correlation_plot_{target}.png")
    plt.close()

    print(f"Saved to: correlation_only_{target}.csv and correlation_plot_{target}.png")

    # Top 10 correlated features (by absolute value)
    top_corr = corr_df.reindex(corr_df['Correlation'].abs().sort_values(ascending=False).index)
    top_10 = top_corr.head(10)
    top_10.to_csv(f"top10_correlated_only_{target}.csv", index=False)

    # Horizontal bar plot of top 10
    plt.figure(figsize=(10, 6))
    plt.barh(top_10['Feature'], top_10['Correlation'],
             color=['green' if c > 0 else 'red' for c in top_10['Correlation']])
    plt.xlabel(f"Correlation with {target}")
    plt.title(f"Top 10 Most Correlated Features with {target}")
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.gca().invert_yaxis()
    plt.grid(True, axis='x', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"top10_correlated_only_{target}.png")
    plt.close()

    print(f"Saved to: top10_correlated_only_{target}.csv and top10_correlated_only_{target}.png")
