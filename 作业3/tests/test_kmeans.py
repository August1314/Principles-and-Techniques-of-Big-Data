import unittest

import numpy as np

from src.kmeans import KMeansClustering


class KMeansClusteringTests(unittest.TestCase):
    def test_kmeans_separates_two_compact_clusters(self) -> None:
        points = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.2],
                [5.0, 5.0],
                [5.2, 4.9],
            ]
        )

        model = KMeansClustering(n_clusters=2, max_iter=20, tol=1e-6, random_state=0)
        labels = model.fit_predict(points)

        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])
        self.assertEqual(model.cluster_centers_.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
