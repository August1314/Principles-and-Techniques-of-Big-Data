import unittest

import numpy as np

from src.dbscan import DBSCANClustering


class DBSCANClusteringTests(unittest.TestCase):
    def test_dbscan_detects_cluster_and_noise(self) -> None:
        points = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.1],
                [0.1, 0.0],
                [5.0, 5.0],
            ]
        )

        model = DBSCANClustering(eps=0.25, min_samples=3)
        labels = model.fit_predict(points)

        self.assertTrue(np.all(labels[:3] == labels[0]))
        self.assertEqual(labels[3], -1)


if __name__ == "__main__":
    unittest.main()
