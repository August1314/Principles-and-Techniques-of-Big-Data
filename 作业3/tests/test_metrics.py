import unittest

import numpy as np

from src.metrics import (
    calinski_harabasz_score_custom,
    clustering_accuracy,
    silhouette_score_custom,
)


class MetricsTests(unittest.TestCase):
    def test_clustering_accuracy_maps_clusters_to_labels(self) -> None:
        true_labels = np.array([0, 0, 1, 1])
        predicted_labels = np.array([1, 1, 0, 0])

        accuracy = clustering_accuracy(true_labels, predicted_labels)

        self.assertAlmostEqual(accuracy, 1.0)

    def test_internal_scores_reward_compact_well_separated_clusters(self) -> None:
        points = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.1],
                [3.0, 3.0],
                [3.2, 3.1],
            ]
        )
        labels = np.array([0, 0, 1, 1])

        silhouette = silhouette_score_custom(points, labels)
        ch_score = calinski_harabasz_score_custom(points, labels)

        self.assertGreater(silhouette, 0.8)
        self.assertGreater(ch_score, 100.0)


if __name__ == "__main__":
    unittest.main()
