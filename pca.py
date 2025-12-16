import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.decomposition import PCA
import pandas as pd

#load dataset and prepare feature matrix
df = pd.read_csv("bank.csv")
X_df = df.drop(columns=['deposit'])
X_df = pd.get_dummies(X_df, drop_first=True)
X = X_df.values

#fit pca to analyse variance 
pca = PCA()
pca.fit(X)

#plot variance contribution of each principal component
variance_ratio = pca.explained_variance_ratio_
plt.figure()
plt.plot(variance_ratio)
plt.xlabel('Component index')
plt.ylabel('Proportion of variance')
plt.title('Variance contribution of principal components')
plt.show()

#plot cumulative variance to assess effective dimensionality
plt.figure()
plt.plot(np.cumsum(variance_ratio))
plt.xlabel('Number of components')
plt.ylabel('Cumulative variance')
plt.title('Cumulative variance explained')
plt.show()


#project data onto first two principal components
pca_2 = PCA(2)
projected = pca_2.fit_transform(X)
plt.figure()
plt.scatter(projected[:, 0], projected[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA projection (PC1 vs PC2)')
plt.show()