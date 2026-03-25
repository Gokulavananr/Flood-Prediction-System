import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("data/flood.csv")

print("Columns:", data.columns)

# ❌ Drop text column
data = data.drop("Disaster Type", axis=1)

# 🎯 Target column
target_column = "occured"

# Features & target
X = data.drop(target_column, axis=1)
y = data[target_column]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained successfully!")