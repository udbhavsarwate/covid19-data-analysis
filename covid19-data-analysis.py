import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load COVID-19 data
df = pd.read_csv("C:\Users\Admin\OneDrive\Documents\covid19-data-analysis.py")

# Inspect the data
print(df.head())
print(df.info())

# Convert date column to datetime
df['last_updated_date'] = pd.to_datetime(df['last_updated_date'])

# Filter data for India
df_India = df[
    (df["location"] == 'India') &
    (df["total_cases"] >= 1000) &
    (df["last_updated_date"] >= "2020-04-08")
]

# Check filtered data
print(df_India.head())
print(df_India[['last_updated_date', 'new_cases', 'new_deaths', 'new_vaccinations']].head())

# Drop NaNs in new_cases
df_India = df_India.dropna(subset=["new_cases"])

# Add rolling average columns
df_India['new_cases_smoothed'] = df_India['new_cases'].rolling(window=7).mean()
df_India['new_deaths_smoothed'] = df_India['new_deaths'].rolling(window=7).mean()

# ========================
# PLOT 1: Daily New Cases
# ========================
plt.figure(figsize=(12, 6))
plt.plot(df_India['last_updated_date'], df_India['new_cases'], label="Daily New Cases", color='orange')
plt.title("COVID-19 Daily New Cases in India")
plt.xlabel("Date")
plt.ylabel("Daily New Cases")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ========================
# PLOT 2: Daily Cases, Deaths, Vaccinations
# ========================
plt.figure(figsize=(14, 8))
plt.plot(df_India['last_updated_date'], df_India['new_cases'], label='New Cases', color='blue')
plt.plot(df_India['last_updated_date'], df_India['new_deaths'], label='New Deaths', color='red')
plt.plot(df_India['last_updated_date'], df_India['new_vaccinations'], label='New Vaccinations', color='green')
plt.title("COVID-19 Daily New Cases, Deaths, Vaccinations in India")
plt.xlabel("Date")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ========================
# PLOT 3: Daily vs 7-Day Average
# ========================
plt.figure(figsize=(14, 10))
plt.plot(df_India['last_updated_date'], df_India['new_cases'], alpha=0.4, label='Daily Cases')
plt.plot(df_India['last_updated_date'], df_India['new_cases_smoothed'], label='7-day Avg Cases', color='blue')
plt.plot(df_India['last_updated_date'], df_India['new_deaths'], alpha=0.4, label='Daily Deaths')
plt.plot(df_India['last_updated_date'], df_India['new_deaths_smoothed'], label='7-day Avg Deaths', color='red')
plt.title("COVID-19 India: Daily vs 7-Day Average")
plt.xlabel("Date")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ========================
# PLOT 4: Compare Countries
# ========================
countries = ['India', 'United States', 'Brazil']
df_multi = df[df['location'].isin(countries)].copy()
df_multi['last_updated_date'] = pd.to_datetime(df_multi['last_updated_date'])

plt.figure(figsize=(14, 8))
for country in countries:
    subset = df_multi[df_multi['location'] == country]
    plt.plot(subset['last_updated_date'], subset['new_cases'].rolling(window=7).mean(), label=country)

plt.title("7-Day Average New Cases: India vs USA vs Brazil")
plt.xlabel("Date")
plt.ylabel("New Cases (7-day avg)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ========================
# PLOT 5: Correlation Heatmap
# ========================
df_India_corr = df_India[['new_cases', 'new_deaths', 'new_vaccinations']].dropna()
plt.figure(figsize=(8, 6))
sns.heatmap(df_India_corr.corr(), annot=True, cmap='coolwarm', square=True)
plt.title("Correlation Matrix - India")
plt.tight_layout()
plt.show()

# Save cleaned data
df_India.to_csv(r"C:\Users\Admin\OneDrive\Documents\covid19-data-analysis.py", index=False)
print("✅ Cleaned India COVID data saved.")