import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pickle

# Load dataset
data = pd.read_csv("data/flood.csv")

print("Columns:", data.columns)

# ❌ Drop text column
data = data.drop("Disaster Type", axis=1)

# 🎯 Target column (classification)
target_column = "occured"

# 🎯 Death target (regression)
y_deaths = data["Total Deaths"]

# Features & targets
X = data.drop(target_column, axis=1)
y = data[target_column]

# Split (use same indices for both models)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 🔹 Classification model (Disaster Prediction)
model = RandomForestClassifier(class_weight="balanced")
model.fit(X_train, y_train)

# 🔹 Regression model (Death Prediction)
death_model = RandomForestRegressor()
death_model.fit(X_train, y_deaths.loc[X_train.index])

# Save models
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(death_model, open("death_model.pkl", "wb"))

print("✅ Both models trained successfully!")