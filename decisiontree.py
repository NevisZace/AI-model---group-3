import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, make_scorer

BankData = pd.read_csv("bank.csv")
X = BankData.drop('deposit', axis=1)
y = BankData['deposit']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
category = X.select_dtypes(include='object').columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), category)
    ],
    remainder='passthrough'
)
X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc  = preprocessor.transform(X_test)
dt_untuned = DecisionTreeClassifier(random_state=42)
dt_untuned.fit(X_train_enc, y_train)
y_pred_untuned = dt_untuned.predict(X_test_enc)

print("Untuned Decision Tree")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_untuned):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_untuned, pos_label='yes'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_untuned, pos_label='yes'):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred_untuned, pos_label='yes'):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_untuned))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_untuned))

max_depths = range(1, 21)
train_f1_scores = []
cv_f1_scores = []
f1_scorer = make_scorer(f1_score, pos_label='yes')

for depth in max_depths:
    dt = DecisionTreeClassifier(random_state=42, max_depth=depth)
    dt.fit(X_train_enc, y_train)
    train_f1_scores.append(f1_score(y_train, dt.predict(X_train_enc), pos_label='yes'))
    cv_score = np.mean(cross_val_score(dt, X_train_enc, y_train, cv=5, scoring=f1_scorer))
    cv_f1_scores.append(cv_score)

#Overfit/underfit
plt.figure(figsize=(8,5))
plt.plot(max_depths, train_f1_scores, label='Training F1', marker='o')
plt.plot(max_depths, cv_f1_scores, label='CV F1', marker='o')
plt.xlabel('max_depth')
plt.ylabel('F1-score')
plt.title('Decision Tree Overfit vs Underfit')
plt.legend()
plt.grid(True)
plt.show()
best_depth = max_depths[np.argmax(cv_f1_scores)]
print(f"Best max_depth from CV: {best_depth}")

dt_tuned = DecisionTreeClassifier(
    random_state=42,
    max_depth=best_depth,
    min_samples_split=50,
    min_samples_leaf=5,
    criterion='entropy',
    max_features=0.9
)
dt_tuned.fit(X_train_enc, y_train)
y_pred_tuned = dt_tuned.predict(X_test_enc)

print("\nTuned Decision Tree")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_tuned):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_tuned, pos_label='yes'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_tuned, pos_label='yes'):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred_tuned, pos_label='yes'):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tuned))
