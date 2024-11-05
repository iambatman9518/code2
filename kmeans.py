import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('creditcard.csv')
df = df.dropna()

X = df[['Feature1', 'Feature2']]

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

df['Cluster'] = kmeans.labels_

plt.scatter(X['Feature1'], X['Feature2'], c=df['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()
plt.title("K-Means Clustering")
plt.show()
