import unittest
import pandas as pd
import numpy as np
from K_MeansApp import load_data, preprocess_data, plot_elbow_method, apply_kmeans

class KmeansAppTest(unittest.TestCase):
    def setUp(self):
        # Load the data
        self.df = load_data('Dataset/Wholesale customers data.csv')
        if self.df is not None:
            # Preprocess the data and unpack the tuple
            self.cleaned_df, self.df_scaled = preprocess_data(self.df)
            if self.df_scaled is not None:
                # Apply K-Means clustering
                self.clustered_df = apply_kmeans(self.df_scaled, self.cleaned_df.copy(), n_clusters=3)
        else:
            self.cleaned_df = None
            self.df_scaled = None
            self.clustered_df = None
        # Debug print to check return values
        print(f"Setup - df_scaled type: {type(self.df_scaled)}, value: {self.df_scaled[:5] if self.df_scaled is not None and hasattr(self.df_scaled, '__getitem__') else self.df_scaled}")

    def test_data_loading(self):
        self.assertIsNotNone(self.df, "Dataset failed to load")
        if self.df is not None:
            self.assertEqual(self.df.shape[0], 440, "Incorrect number of rows")
            self.assertEqual(self.df.shape[1], 8, "Incorrect number of columns")

    def test_data_preprocessing(self):
        if self.df_scaled is not None:
            self.assertEqual(self.df_scaled.shape[1], 6, "Incorrect number of features")
            self.assertAlmostEqual(np.mean(self.df_scaled), 0, places=5, msg="Scaled data should have zero mean")
            self.assertAlmostEqual(np.std(self.df_scaled), 1, places=5, msg="Scaled data should have unit variance")
        else:
            self.fail("Preprocessing failed to produce a valid scaled array")

    def test_kmeans_clustering(self):
        if self.clustered_df is not None:
            self.assertIn('Cluster', self.clustered_df.columns, "Cluster column should be added")
            self.assertEqual(len(set(self.clustered_df['Cluster'])), 3, "Should have 3 clusters")
        else:
            self.fail("K-Means clustering failed to produce a valid result")

if __name__ == '__main__':
    unittest.main()