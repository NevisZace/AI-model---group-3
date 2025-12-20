import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset (ensure bank.csv is in the repository)
BankData = pd.read_csv("bank.csv")
X = BankData.drop('deposit', axis=1)
y = BankData['deposit']

# Encode categorical variables as per group style
category = X.select_dtypes(include='object').columns
label = LabelEncoder()
for col in category:
    X[col] = label.fit_transform(X[col])
y = y.map({'no': 0, 'yes': 1})

# Standard 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Random Forest Model (Your specific contribution)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
rf_model.fit(X_train, y_train)

# Output results for the report
y_pred = rf_model.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Confusion Matrix (Required for the 8-page report visuals)
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
plt.title('Random Forest Confusion Matrix')
plt.show()
