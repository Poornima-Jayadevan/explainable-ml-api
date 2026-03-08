import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import classification_report


#load the excel file
df = pd.read_csv("credit_risk_dataset.csv")

print(df.shape)
print(df.head())
print(df['loan_status'].value_counts())



# Define features and target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

#categorical columns
categorical_cols = [
    'person_home_ownership', 
    'loan_intent', 
    'loan_grade', 
    'cb_person_default_on_file'
]

#one-hot encoding
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
joblib.dump(list(X_encoded.columns), "feature_names.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Save CSVs
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)


# Initialize model
model = RandomForestClassifier(n_estimators=100, random_state=42)


# Train
model.fit(X_train, y_train)


# Save model
joblib.dump(model, "risk_model.pkl")


# Optional: Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

