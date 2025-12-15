import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

#load dataset and define data/target
BankData = pd.read_csv("bank.csv")
X = BankData.drop(columns=["deposit"])
y = BankData["deposit"]

#train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#predict majority class
class_counts = y_train.value_counts()
majority_class = class_counts.index[0]
y_pred = []
for i in range(len(y_test)):
    y_pred.append(majority_class)
    
  #evaluation metrics  
print("Majority class:", majority_class)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Precision: %.2f' % precision_score(y_test, y_pred, pos_label='yes',zero_division=0))
print('Recall: %.2f' % recall_score(y_test, y_pred, pos_label='yes',zero_division=0))
print('F1: %.2f' % f1_score(y_test, y_pred, pos_label='yes',zero_division=0))

#plot confusion matrix and normalised confusion matrix
labels = ['no', 'yes']

def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.05)
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm = confusion_matrix(y_test, y_pred, labels=labels)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)

plt.figure()
plot_confusion_matrix(cm, labels, title='Baseline confusion matrix')
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)

plt.figure()
plot_confusion_matrix(cm_normalized, labels, title='Normalized confusion matrix')

plt.show()
