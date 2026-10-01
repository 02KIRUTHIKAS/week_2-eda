"""
Week 2 - Exploratory Data Analysis and Visualization
Dataset: Breast Cancer Wisconsin (Diagnostic) - UCI ML Repository
Libraries: Pandas, Matplotlib, Seaborn
Run:  python eda_breast_cancer.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

sns.set_theme(style="whitegrid", palette="Set2")
FIG = "figures/"
PAL = {"Benign": "#66c2a5", "Malignant": "#fc8d62"}  # same colours in every chart

# ------------------------------------------------------------------
# 1. Load the dataset
# ------------------------------------------------------------------
raw = load_breast_cancer(as_frame=True)
df = raw.frame.copy()
# In the raw data target: 0 = malignant, 1 = benign. Make it readable.
df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})
df = df.drop(columns="target")

print("Shape:", df.shape)
print(df.dtypes.value_counts())
print("Missing values:", df.isna().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print(df["diagnosis"].value_counts())

# ------------------------------------------------------------------
# 2. Basic statistics
# ------------------------------------------------------------------
mean_cols = [c for c in df.columns if c.startswith("mean ")]
desc = df[mean_cols].describe().T[["mean", "std", "min", "50%", "max"]].round(3)
desc["skew"] = df[mean_cols].skew().round(3)
print(desc)
desc.to_csv("summary_statistics.csv")

# ------------------------------------------------------------------
# 3. Data transformation / aggregation
# ------------------------------------------------------------------
# (a) group means by diagnosis
grp = df.groupby("diagnosis")[mean_cols].mean().T.round(3)
grp["ratio (M/B)"] = (grp["Malignant"] / grp["Benign"]).round(2)
print(grp.sort_values("ratio (M/B)", ascending=False))

# (b) outlier count per feature using the IQR rule
def iqr_outliers(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
outliers = df[mean_cols].apply(iqr_outliers).sort_values(ascending=False)
print(outliers)

# (c) new feature: size category from mean radius quartiles
df["size_group"] = pd.qcut(df["mean radius"], 4,
                           labels=["Small", "Medium", "Large", "Very large"])
rate = (df.groupby("size_group", observed=True)["diagnosis"]
          .apply(lambda s: (s == "Malignant").mean() * 100).round(1))
print(rate)

# ------------------------------------------------------------------
# 4. Visualizations
# ------------------------------------------------------------------
# Fig 1: class balance
fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(data=df, x="diagnosis", order=["Benign", "Malignant"], hue="diagnosis", palette=PAL, legend=False, ax=ax)
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
ax.set_title("Figure 1: Distribution of Diagnosis")
ax.set_xlabel("Diagnosis"); ax.set_ylabel("Number of patients")
plt.tight_layout(); plt.savefig(FIG + "fig1_class_balance.png", dpi=150); plt.close()

# Fig 2: histograms of key features by diagnosis
feats = ["mean radius", "mean texture", "mean perimeter", "mean area"]
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, f in zip(axes.ravel(), feats):
    sns.histplot(data=df, x=f, hue="diagnosis", palette=PAL, kde=True, bins=30, ax=ax)
    ax.set_title(f"Distribution of {f}")
    ax.set_xlabel(f); ax.set_ylabel("Count")
fig.suptitle("Figure 2: Feature distributions by diagnosis", y=1.01)
plt.tight_layout(); plt.savefig(FIG + "fig2_histograms.png", dpi=150, bbox_inches="tight"); plt.close()

# Fig 3: boxplots (outliers + group differences)
fig, axes = plt.subplots(1, 4, figsize=(13, 4.5))
for ax, f in zip(axes, ["mean radius", "mean area", "mean concavity", "mean smoothness"]):
    sns.boxplot(data=df, x="diagnosis", y=f, order=["Benign", "Malignant"], hue="diagnosis", palette=PAL, legend=False, ax=ax)
    ax.set_title(f); ax.set_xlabel("Diagnosis"); ax.set_ylabel(f)
fig.suptitle("Figure 3: Boxplots by diagnosis", y=1.02)
plt.tight_layout(); plt.savefig(FIG + "fig3_boxplots.png", dpi=150, bbox_inches="tight"); plt.close()

# Fig 4: correlation heatmap
corr = df[mean_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            annot_kws={"size": 7}, cbar_kws={"label": "Pearson correlation"}, ax=ax)
ax.set_title("Figure 4: Correlation heatmap of mean features")
plt.tight_layout(); plt.savefig(FIG + "fig4_heatmap.png", dpi=150); plt.close()

# Fig 5: scatter radius vs area / concavity
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.scatterplot(data=df, x="mean radius", y="mean area", hue="diagnosis", palette=PAL, alpha=0.7, ax=axes[0])
axes[0].set_title("Mean radius vs mean area")
sns.scatterplot(data=df, x="mean concavity", y="mean concave points", hue="diagnosis", palette=PAL, alpha=0.7, ax=axes[1])
axes[1].set_title("Mean concavity vs mean concave points")
fig.suptitle("Figure 5: Relationships between features", y=1.02)
plt.tight_layout(); plt.savefig(FIG + "fig5_scatter.png", dpi=150, bbox_inches="tight"); plt.close()

# Fig 6: malignancy rate by size group
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=rate.index, y=rate.values, ax=ax)
for i, v in enumerate(rate.values):
    ax.text(i, v + 1, f"{v}%", ha="center")
ax.set_title("Figure 6: Malignancy rate by tumour size group")
ax.set_xlabel("Mean radius quartile"); ax.set_ylabel("% malignant"); ax.set_ylim(0, 110)
plt.tight_layout(); plt.savefig(FIG + "fig6_size_rate.png", dpi=150); plt.close()

# Fig 7: correlation of each feature with malignancy
df["is_malignant"] = (df["diagnosis"] == "Malignant").astype(int)
tc = df[mean_cols + ["is_malignant"]].corr()["is_malignant"].drop("is_malignant").sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
tc.plot(kind="barh", ax=ax, color=["#66c2a5" if v < 0 else "#fc8d62" for v in tc])
ax.set_title("Figure 7: Correlation of each feature with malignancy")
ax.set_xlabel("Pearson correlation with is_malignant"); ax.set_ylabel("Feature")
plt.tight_layout(); plt.savefig(FIG + "fig7_target_corr.png", dpi=150); plt.close()
print(tc.round(2))

# Highly correlated pairs
pairs = (corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
             .stack().reset_index())
pairs.columns = ["f1", "f2", "r"]
print(pairs[pairs.r.abs() > 0.9].sort_values("r", ascending=False).round(3))
