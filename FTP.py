import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt


airlines = pd.read_csv("C:/Users/tarun/Desktop/US Airline Flight Routes and Fares 1993-2024.csv", low_memory=False)
print(airlines.head())

print(airlines.info())

print(airlines.describe())

missing_values = airlines.isnull().sum()
print(missing_values)

df_cleaned = airlines.dropna()

df_cleaned_columns = airlines.dropna(axis=1)

print(df_cleaned.shape)
print(df_cleaned_columns.shape)

print(df_cleaned.info())

duplicates = airlines.duplicated().sum()
print(f'Duplicates: {duplicates}')

airlines.drop(['tbl', 'citymarketid_1', 'citymarketid_2', 'airportid_1', 'airportid_2', 'tbl1apk'], axis=1, inplace=True)
target = 'fare'

categorical_columns = airlines.select_dtypes(include=['object']).columns
print("Categorical Columns:", categorical_columns)

df_encoded = pd.get_dummies(airlines, columns=categorical_columns, drop_first=True)

X = df_encoded.drop(columns=[target])
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=importances.head(10))
plt.title('Top 10 Feature Importances from Random Forest')
plt.show()

cov_matrix = df_encoded.cov()
sns.heatmap(cov_matrix, cmap='coolwarm', annot=False)
plt.title('Covariance Matrix Heatmap')
plt.show()