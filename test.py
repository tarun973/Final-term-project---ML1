import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# Replace 'your_dataset.csv' with your file
data = pd.read_csv('your_dataset.csv')

# 1. Preprocessing the Dataset for Clustering
# Remove non-numerical or irrelevant features for clustering
numerical_data = data.select_dtypes(include=[np.number])  # Only numerical columns
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)


# 2. K-Means or K-Means++ Algorithm
def kmeans_clustering(data):
    print("\n--- K-Means Clustering ---")

    # Determine optimal number of clusters using Silhouette Analysis
    silhouette_scores = []
    k_values = range(2, 11)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++')
        cluster_labels = kmeans.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, cluster_labels))

    # Plot Silhouette Analysis
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.title("Silhouette Analysis for K-Means")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid()
    plt.show()

    # Fit KMeans with optimal k (based on Silhouette Analysis)
    optimal_k = k_values[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k}")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, init='k-means++')
    cluster_labels = kmeans.fit_predict(data)

    # Add cluster labels to the original dataset
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = cluster_labels

    # Within-Cluster Variation Plot
    wcss = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++')
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(k_values, wcss, marker='o')
    plt.title("Within-Cluster Variation Plot (WCSS)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS")
    plt.grid()
    plt.show()

    return data_with_clusters


# Apply K-Means Clustering
clustered_data_kmeans = kmeans_clustering(scaled_data)


# 3. DBSCAN Algorithm
def dbscan_clustering(data):
    print("\n--- DBSCAN Clustering ---")

    # Find the optimal epsilon using k-distance plot
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)
    distances = np.sort(distances[:, -1])

    # Plot k-distance graph
    plt.figure(figsize=(8, 6))
    plt.plot(distances)
    plt.title("k-Distance Graph for DBSCAN")
    plt.xlabel("Data Points")
    plt.ylabel("Distance to 5th Nearest Neighbor")
    plt.grid()
    plt.show()

    # Choose epsilon (elbow point) based on the plot
    epsilon = float(input("Enter the optimal epsilon value based on the plot: "))

    # Fit DBSCAN
    dbscan = DBSCAN(eps=epsilon, min_samples=5, metric='euclidean')
    cluster_labels = dbscan.fit_predict(data)
    print(f"Number of clusters formed: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")

    # Add cluster labels to the original dataset
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = cluster_labels

    return data_with_clusters


# Apply DBSCAN Clustering
clustered_data_dbscan = dbscan_clustering(scaled_data)


# 4. Apriori Algorithm for Association Rule Mining
def association_rule_mining(data, min_support=0.01, min_confidence=0.5):
    print("\n--- Association Rule Mining using Apriori Algorithm ---")

    # Preprocessing for Apriori: Convert dataset into transaction format (binary encoding)
    # Select categorical columns and one-hot encode them
    categorical_data = data.select_dtypes(include=['object', 'category'])
    one_hot_encoded_data = pd.get_dummies(categorical_data, drop_first=True)

    # Apply Apriori algorithm
    frequent_itemsets = apriori(one_hot_encoded_data, min_support=min_support, use_colnames=True)
    print("\nFrequent Itemsets:")
    print(frequent_itemsets)

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    print("\nAssociation Rules:")
    print(rules)

    return frequent_itemsets, rules


# Apply Apriori Algorithm
frequent_itemsets, association_rules = association_rule_mining(data, min_support=0.01, min_confidence=0.5)
