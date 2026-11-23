import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

df = pd.read_csv("combined_cars_clean.csv")  # لازم يكون موجود في نفس مجلد التشغيل (خرج من Part 3)

print("="*70)
print("4.1 NUMERICAL SUMMARY (Price)")
print("="*70)
print(df["price"].describe())
print()

# ----------------------------------------------------------------------
# 4.2 Price Distribution (+ Log Price)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(df["price"], bins=80, color="#4C72B0", edgecolor="white")
axes[0].set_title("Price Distribution (Raw)")
axes[0].set_xlabel("Price (£)")
axes[0].set_ylabel("Number of Listings")

log_price = np.log1p(df["price"])
axes[1].hist(log_price, bins=80, color="#55A868", edgecolor="white")
axes[1].set_title("Price Distribution (Log-transformed)")
axes[1].set_xlabel("log(1 + Price)")
axes[1].set_ylabel("Number of Listings")

plt.tight_layout()
plt.savefig("01_price_distribution.png")
plt.show()
plt.close()

skew_raw = df["price"].skew()
skew_log = log_price.skew()
print("="*70)
print("4.2 PRICE DISTRIBUTION")
print("="*70)
print(f"Raw price skewness: {skew_raw:.3f}")
print(f"Log price skewness: {skew_log:.3f}")
print()

# ----------------------------------------------------------------------
# 4.3 Price vs Mileage
# ----------------------------------------------------------------------
plt.figure(figsize=(8, 6))
sample = df.sample(n=min(15000, len(df)), random_state=42)  # sample for readability
plt.scatter(sample["mileage"], sample["price"], alpha=0.15, s=8, color="#4C72B0")

z = np.polyfit(df["mileage"], df["price"], 1)
p = np.poly1d(z)
x_line = np.linspace(df["mileage"].min(), df["mileage"].max(), 100)
plt.plot(x_line, p(x_line), color="red", linewidth=2, label="Trend line")

plt.title("Price vs Mileage")
plt.xlabel("Mileage")
plt.ylabel("Price (£)")
plt.legend()
plt.tight_layout()
plt.savefig("02_price_vs_mileage.png")
plt.show()
plt.close()

corr_mileage = df["price"].corr(df["mileage"])
print("="*70)
print("4.3 PRICE vs MILEAGE")
print("="*70)
print(f"Correlation (price, mileage): {corr_mileage:.3f}")
print()

# ----------------------------------------------------------------------
# 4.4 Average Price by Year (boxplot, years with enough data)
# ----------------------------------------------------------------------
year_counts = df["year"].value_counts()
valid_years = sorted(year_counts[year_counts >= 30].index)
df_year = df[df["year"].isin(valid_years)]

plt.figure(figsize=(12, 6))
order = sorted(valid_years)
sns.boxplot(data=df_year, x="year", y="price", order=order, color="#4C72B0", showfliers=False)
plt.title("Price by Year (boxplot, outlier points hidden for readability)")
plt.xlabel("Year")
plt.ylabel("Price (£)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("03_price_by_year.png")
plt.show()
plt.close()

year_stats = df.groupby("year")["price"].agg(["mean", "median", "count"]).sort_index()
print("="*70)
print("4.4 PRICE BY YEAR (10 most recent model-years present in the cleaned dataset)")
print("="*70)
print(year_stats.tail(10))
print()
corr_year = df["price"].corr(df["year"])
print(f"Correlation (price, year): {corr_year:.3f}")
print()

# ----------------------------------------------------------------------
# 4.5 Average Price by Manufacturer (+ median table)
# ----------------------------------------------------------------------
make_stats = df.groupby("Make")["price"].agg(["mean", "median", "count"]).sort_values("mean", ascending=False)

plt.figure(figsize=(10, 7))
make_stats["mean"].sort_values().plot(kind="barh", color="#4C72B0")
plt.title("Average Price by Manufacturer")
plt.xlabel("Average Price (£)")
plt.ylabel("Manufacturer")
plt.tight_layout()
plt.savefig("04_avg_price_by_manufacturer.png")
plt.show()
plt.close()

print("="*70)
print("4.5 AVERAGE PRICE BY MANUFACTURER (Mean vs Median, sorted by Mean)")
print("="*70)
print(make_stats)
print()

# ----------------------------------------------------------------------
# 4.6 Price by Transmission
# ----------------------------------------------------------------------
trans_counts = df["transmission"].value_counts()
plt.figure(figsize=(8, 6))
order = trans_counts.index.tolist()
sns.boxplot(data=df, x="transmission", y="price", order=order, color="#55A868", showfliers=False)
plt.title("Price by Transmission Type")
plt.xlabel("Transmission")
plt.ylabel("Price (£)")
plt.tight_layout()
plt.savefig("05_price_by_transmission.png")
plt.show()
plt.close()

print("="*70)
print("4.6 PRICE BY TRANSMISSION")
print("="*70)
print(trans_counts)
print()
print(df.groupby("transmission")["price"].median().sort_values(ascending=False))
print()

# ----------------------------------------------------------------------
# 4.7 Price by Fuel Type
# ----------------------------------------------------------------------
fuel_counts = df["fuelType"].value_counts()
plt.figure(figsize=(8, 6))
order = fuel_counts.index.tolist()
sns.boxplot(data=df, x="fuelType", y="price", order=order, color="#C44E52", showfliers=False)
plt.title("Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Price (£)")
plt.tight_layout()
plt.savefig("06_price_by_fueltype.png")
plt.show()
plt.close()

print("="*70)
print("4.7 PRICE BY FUEL TYPE")
print("="*70)
print(fuel_counts)
print()
print(df.groupby("fuelType")["price"].median().sort_values(ascending=False))
print()

# ----------------------------------------------------------------------
# 4.8 Price vs Engine Size
# ----------------------------------------------------------------------
df_eng = df.dropna(subset=["engineSize"])
plt.figure(figsize=(8, 6))
sample_eng = df_eng.sample(n=min(15000, len(df_eng)), random_state=42)
plt.scatter(sample_eng["engineSize"], sample_eng["price"], alpha=0.15, s=8, color="#8172B2")
plt.title("Price vs Engine Size")
plt.xlabel("Engine Size (L)")
plt.ylabel("Price (£)")
plt.tight_layout()
plt.savefig("07_price_vs_enginesize.png")
plt.show()
plt.close()

corr_engine = df_eng["price"].corr(df_eng["engineSize"])
print("="*70)
print("4.8 PRICE vs ENGINE SIZE")
print("="*70)
print(f"Correlation (price, engineSize): {corr_engine:.3f}")
print(f"(computed on {len(df_eng)} rows with non-missing engineSize; {df['engineSize'].isna().sum()} rows excluded)")
print()

# ----------------------------------------------------------------------
# 4.9 Correlation Heatmap
# ----------------------------------------------------------------------
numeric_cols = ["year", "price", "mileage", "tax", "mpg", "engineSize"]
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5)
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig("08_correlation_heatmap.png")
plt.show()
plt.close()

print("="*70)
print("4.9 CORRELATION MATRIX")
print("="*70)
print(corr_matrix.round(3))
print()

print("All 8 figures saved to the current working directory.")
