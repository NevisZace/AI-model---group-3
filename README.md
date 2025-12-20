# Bank Marketing – Term Deposit Prediction

## Dataset
The dataset used in this project is the Bank Marketing dataset, obtained from Kaggle:  
https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data

## Description
This project investigates a binary classification problem using the Bank Marketing dataset.  
The objective is to predict whether a customer will subscribe to a term deposit (deposit = yes/no) based on demographic and behavioural features.

Both supervised and unsupervised learning techniques from scikit-learn are applied and compared.

## Repository Structure

- main.py – Initial exploratory data analysis (EDA)
- baseline_majority.py – Majority class baseline model
- decisiontree.py – Decision Tree classifier with cross-validation, overfitting analysis, and GridSearchCV
- random_forest.py – Random Forest classifier
- linear_regression.py – Linear regression model (later rejected from final analysis)
- k_means_clustering.py – Unsupervised clustering analysis using k-means
- pca.py – Principal Component Analysis (PCA)
- bank.csv – Dataset used for analysis

## Notes
- A stratified train–test split and cross-validation were used consistently across models.
- Linear regression was explored initially but excluded from the final evaluation due to modelling assumptions.
