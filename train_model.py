import pandas as pd
#from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json

# Load the Iris dataset
df = pd.read_csv("Data/iris.data", header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Split the data
X = df.iloc[:, :-1]  # Features (first four columns)
y = df.iloc[:, -1]  # Target variable (last column)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save results to a JSON file
results = {
    "accuracy": accuracy, "feature_importances": list(model.feature_importances_)
}
with open("results.json", "w") as f:
    json.dump(results, f)

print("Model training completed. Results saved to results.json.")