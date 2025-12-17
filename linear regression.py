import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

BankData = pd.read_csv("bank.csv")
X = BankData.drop('deposit', axis=1)
y = BankData['deposit']

#Encode categorical variables
category = X.select_dtypes(include='object').columns
label = LabelEncoder()
for col in category:
    X[col] = label.fit_transform(X[col])
y = y.map({'no': 0, 'yes': 1})

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_continuous = lr_model.predict(X_test)
y_pred = (y_pred_continuous >= 0.5).astype(int)

print("Performance")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['no', 'yes']))
