#replaces earlier draft with tuned Random Forest:
#one-hot encoding 
#5-fold CV tuning over n_estimators
#F1-score selection
#final evaluation on a 20% holdout test set

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, make_scorer
)

#load dataset
BankData = pd.read_csv("bank.csv")
X = BankData.drop("deposit", axis=1)
y = BankData["deposit"]

#holdout test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

#preprocess: one-hot encoding
categorical_cols = X.select_dtypes(include="object").columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

f1_scorer = make_scorer(f1_score, pos_label="yes")

#try different trees
nums = []
cv_f1 = []
train_f1 = []
for n in range(10, 201, 5):
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("rf", RandomForestClassifier(
            n_estimators=n,
            criterion="entropy",
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    #cross validation on training data

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring=f1_scorer,
        return_train_score=True
    )

    nums.append(n)
    cv_f1.append(scores["test_score"].mean())
    train_f1.append(scores["train_score"].mean())

best_index = int(np.argmax(cv_f1))
best_n = nums[best_index]
print("Best n_estimators from 5-fold CV (on training set):", best_n)
print("Best mean CV F1:", cv_f1[best_index])

#plot training vs CV F1
plt.figure()
plt.plot(nums, train_f1, label="Mean Training F1 (CV)")
plt.plot(nums, cv_f1, label="Mean CV F1 (5-fold)")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("F1-score")
plt.title("Random Forest: Training vs 5-fold CV F1-score")
plt.grid(True)
plt.legend()
plt.show()

final_model = Pipeline([
    ("preprocess", preprocessor),
    ("rf", RandomForestClassifier(
        n_estimators=best_n,
        criterion="entropy",
        random_state=42,
        n_jobs=-1
    ))
])

final_model.fit(X_train, y_train)
y_pred_test = final_model.predict(X_test)

print("\nRandom Forest (tuned with 5-fold CV, tested once on holdout)")
print("Chosen n_estimators:", best_n)
print("Holdout Test Accuracy:", accuracy_score(y_test, y_pred_test) * 100, "%")
print("\nEvaluation Metrics (Holdout Test Set)")
print("Precision:", precision_score(y_test, y_pred_test, pos_label="yes", zero_division=0))
print("Recall:", recall_score(y_test, y_pred_test, pos_label="yes", zero_division=0))
print("F1-score:", f1_score(y_test, y_pred_test, pos_label="yes", zero_division=0))

#confusion matrix
cm = confusion_matrix(y_test, y_pred_test, labels=["no", "yes"])
print("\nConfusion Matrix (labels: no, yes):")
print(cm)
plt.figure()
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Random Forest Confusion Matrix")
plt.colorbar(fraction=0.05)
plt.xticks([0, 1], ["no", "yes"], rotation=45)
plt.yticks([0, 1], ["no", "yes"])
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.show()
