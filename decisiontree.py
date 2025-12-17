import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

BankData = pd.read_csv("bank.csv")
X = BankData.drop('deposit', axis=1)
y = BankData['deposit']

#Encode categorical variables
category = X.select_dtypes(include='object').columns
label = LabelEncoder()
for col in category:
    X[col] = label.fit_transform(X[col])

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

dt_untuned = DecisionTreeClassifier(random_state=42)
dt_untuned.fit(X_train, y_train)
y_pred_untuned = dt_untuned.predict(X_test)

print("Untuned Decision Tree")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_untuned):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_untuned, pos_label='yes'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_untuned, pos_label='yes'):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred_untuned, pos_label='yes'):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_untuned))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_untuned))

dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=9,            
    min_samples_split=59,    
    min_samples_leaf=6,
    criterion='entropy',
    max_features=0.9
)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

print("Tuned Decision Tree")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, pos_label='yes'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, pos_label='yes'):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred, pos_label='yes'):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))