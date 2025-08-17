
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

#  Load dataset 
df = pd.read_csv("D:\Diabetes_Prediction_System_API_Frontend\diabetes.csv")

X = df.drop("Outcome", axis=1)   # Features
y = df["Outcome"]                # Target

# Split dataset 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#  Scale features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models 
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42)
}

best_model = None
best_score = 0
results = {}

# Train & evaluate models 
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Track best model
    if acc > best_score:
        best_score = acc
        best_model = model
        best_name = name

 # Save best model 
print(f"\nBest Model: {best_name} with Accuracy: {best_score:.4f}")
joblib.dump(best_model, "best_diabetes_model.joblib")
joblib.dump(scaler, "scaler.joblib")  # save scaler for inference

