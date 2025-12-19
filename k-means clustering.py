import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#load dataset and standardise features 
df = pd.read_csv("bank.csv")
X_df = df.drop(columns=["deposit"])
X_df = pd.get_dummies(X_df, drop_first=True)
X = X_df.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_2 = PCA(n_components=2, random_state=42)
projected = pca_2.fit_transform(X_scaled)

# K-means clustering with k=3
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df["cluster"] = clusters

#visualise clusters in PCA 
plt.figure()
plt.scatter(projected[:, 0], projected[:, 1],c=clusters)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means clusters on PCA")
plt.show()

df["deposit_num"] = (df["deposit"] == "yes").astype(int)
print("Cluster sizes:")
print(df["cluster"].value_counts().sort_index())
print("\nDeposit subscription rate per cluster:")
deposit_rates = df.groupby("cluster")["deposit_num"].mean().sort_index()
print(deposit_rates)

#plot deposit subscription rate per cluster
plt.figure()
plt.bar(deposit_rates.index.astype(str), deposit_rates.values)
plt.xlabel("Cluster")
plt.ylabel("Deposit subscription rate")
plt.title("Deposit subscription rate per cluster")
plt.show()