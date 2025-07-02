import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Create a function to load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df

# Function to preprocess the data
def preprocess_data(df):
    if df is None:
        print("Error: No data provided for preprocessing.")
        return None, None

    numerical_columns = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    # First step in preprocessing is removing the outlier using the IQR method
    def remove_outliers(df, columns):
        df_cleaned = df.copy()
        for column in columns:
            Q1 = df_cleaned[column].quantile(0.25)
            Q3 = df_cleaned[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Keep rows where the column value is within bounds
            df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]
        print(f"Original dataset size: {len(df)}, After outlier removal: {len(df_cleaned)}")
        return df_cleaned

    # Remove outliers
    df_cleaned = remove_outliers(df, numerical_columns)

    # Select numerical features for clustering
    X = df_cleaned[numerical_columns]

    # Standardize the dataset
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(X)
    return df_cleaned, df_scaled

# Function to determine optimal number of clusters using Elbow Method
def plot_elbow_method(df_scaled):
    if df_scaled is None:
        print("Error: No scaled data provided for Elbow Method.")
        return

    inertia = []
    k_values = range(2, 10)
    for k in k_values:
        elb = KMeans(n_clusters=k, random_state=42)
        elb.fit(df_scaled)
        inertia.append(elb.inertia_)

    # Plot the curve
    plt.plot(k_values, inertia, marker='o', color='green')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True)
    plt.savefig('elbow_plot.png')
    plt.show()

    # Identify the optimal K using the "elbow" method
    diff = np.diff(inertia)
    k_optimal = k_values[np.argmin(diff) + 1]

    print(f"\n Optimal number of clusters (K) is: {k_optimal}")

# Function to apply K-Means and visualize clusters
def apply_kmeans(df_scaled, df_filtered, n_clusters=3):
    # Apply K-Means
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = model.fit_predict(df_scaled)
    df_filtered = df_filtered.copy()
    df_filtered['Cluster'] = clusters
    # Reduce dimensions for visualization using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df_scaled)

    # Visualize clusters
    plt.figure(figsize=(8, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'K-Means Clustering (k={n_clusters})')
    plt.savefig('cluster_plot.png')
    plt.show()

    return df_filtered

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    file_path = 'Dataset/Wholesale customers data.csv'
    df = load_data(file_path)
    df, df_scaled = preprocess_data(df)

    # Plot Elbow Method to choose k
    plot_elbow_method(df_scaled)

    # Apply K-Means with chosen number of clusters (e.g., k=3 based on Elbow Method)
    data = apply_kmeans(df_scaled, df, n_clusters=3)

    # Save clustered data
    data.to_csv('clustered_wholesale_data.csv', index=False)
    print("Clustering complete. Results saved to 'clustered_wholesale_data.csv'.")





