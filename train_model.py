import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "student_data.csv")

# -----------------------------
# Load dataset
# -----------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("❌ student_data.csv not found")

df = pd.read_csv(DATA_PATH)
print("✅ Dataset loaded")

# -----------------------------
# Target column
# -----------------------------
TARGET = "writing score"
y = df[TARGET]
X = df.drop(TARGET, axis=1)

# -----------------------------
# One-hot encoding
# -----------------------------
X = pd.get_dummies(X, drop_first=True)

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Train model
# -----------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# -----------------------------
# Metrics
# -----------------------------
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# -----------------------------
# Save everything
# -----------------------------
joblib.dump(model, "student_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")
joblib.dump(r2, "r2_score.pkl")
joblib.dump(mae, "mae.pkl")

print("🎯 Model trained successfully")
print(f"R² Score: {r2:.3f}")
print(f"MAE: {mae:.2f}")
print("📦 All files saved")
