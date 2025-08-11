import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Create folder for model and metrics
os.makedirs("artifacts", exist_ok=True)

# Load data
df = pd.read_csv("data/Student_Performance.csv")

# Encode categorical variable
le = LabelEncoder()
df['Extracurricular Activities'] = le.fit_transform(df['Extracurricular Activities'])

# Split data
X = df.drop('Performance Index', axis=1)
y = df['Performance Index']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("R2 Score:", r2)
print("Mean Absolute Error:", mae)

# Save model and metrics
joblib.dump(model, "artifacts/model.pkl")
with open("artifacts/metrics.txt", "w") as f:
    f.write(f"R2 Score: {r2}\n")
    f.write(f"Mean Absolute Error: {mae}\n")
