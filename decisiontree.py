import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, make_scorer
)

#Load dataset
BankData = pd.read_csv("bank.csv")
X = BankData.drop('deposit', axis=1)
y = BankData['deposit']

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#Preprocessing
category = X.select_dtypes(include='object').columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), category)
    ],
    remainder='passthrough'
)
f1_scorer = make_scorer(f1_score, pos_label='yes')

#Untuned decision tree 
untuned_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('dt', DecisionTreeClassifier(random_state=42))
])

cv_scores_untuned = cross_val_score(
    untuned_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring=f1_scorer
)

print("Untuned Decision Tree (5-fold CV on training data)")
print(f"Mean CV F1: {cv_scores_untuned.mean():.4f} ± {cv_scores_untuned.std():.4f}")

#Overfitting vs underfitting
max_depths = range(1, 21)
train_f1 = []
cv_f1 = []

for depth in max_depths:
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('dt', DecisionTreeClassifier(
            random_state=42,
            max_depth=depth
        ))
    ])

    #Train F1
    pipeline.fit(X_train, y_train)
    train_pred = pipeline.predict(X_train)
    train_f1.append(f1_score(y_train, train_pred, pos_label='yes'))
    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring=f1_scorer
    )
    cv_f1.append(scores.mean())
best_depth = max_depths[np.argmax(cv_f1)]
print(f"\nBest max_depth from CV: {best_depth}")

#Plot graph
plt.figure(figsize=(8, 5))
plt.plot(max_depths, train_f1, marker='o', label='Training F1')
plt.plot(max_depths, cv_f1, marker='o', label='CV F1')
plt.xlabel('max_depth')
plt.ylabel('F1-score')
plt.title('Decision Tree Overfit vs Underfit')
plt.legend()
plt.grid(True)
plt.show()

# Hyperparameter tuning with GridSearchCV
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('dt', DecisionTreeClassifier(random_state=42))
])

param_grid = {
    'dt__max_depth': [1, 3, 5, 7, 9, 11, 13, 15, None],
    'dt__min_samples_leaf': [1, 2, 5, 10, 20],
    'dt__min_samples_split': [2, 5, 10, 20, 40, 60, 80, 100],
    'dt__criterion': ['gini', 'entropy', 'log_loss']
}

grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    scoring=f1_scorer,
    cv=5,
    n_jobs=-1
)
grid.fit(X_train, y_train)
print("\nBest parameters from GridSearchCV:")
print(grid.best_params_)

#Tuned decision tree
best_model = grid.best_estimator_
y_pred_tuned = best_model.predict(X_test)
print("\nTuned Decision Tree (Test Set Evaluation)")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_tuned):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_tuned, pos_label='yes'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_tuned, pos_label='yes'):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred_tuned, pos_label='yes'):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tuned))