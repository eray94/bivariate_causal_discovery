import numpy as np
import torch

class HilbertSchmidtInformationCriterion:
    @staticmethod
    def rbf_dot(data: np.ndarray, degree: float) -> np.ndarray:
        """RBF kernel dot operation.

                Args:
                    data (np.ndarray): Data.
                    degree (float):
                Returns:
                    np.ndarray: RBF dot.
        """

        size = data.shape

        g = np.sum(data * data, 1).reshape(size[0], 1)
        h = np.sum(data * data, 1).reshape(size[0], 1)

        q = np.tile(g, (1, size[0]))
        r = np.tile(h.T, (size[0], 1))

        h = q + r - 2 * np.dot(data, data.T)

        h = np.exp(-h / 2 / (degree ** 2))

        return h

    @staticmethod
    def get_width(data: np.ndarray, n: int) -> float:
        """Get width.

                Args:
                    data (np.ndarray): Array.
                    n (int): size.
                Returns:
                    float: Width.
        """

        data_median = data

        g = np.sum(data_median * data_median, 1).reshape(n, 1)
        q = np.tile(g, (1, n))
        r = np.tile(g.T, (n, 1))

        dists = q + r - 2 * np.dot(data_median, data_median.T)
        dists = dists - np.tril(dists)
        dists = dists.reshape(n ** 2, 1)

        width = np.sqrt(0.5 * np.median(dists[dists > 0]))

        return width

    @staticmethod
    def hsic_score(u: torch.autograd.Variable, v:torch.autograd.Variable) -> float:
        """Calculate independence score.

                Args:
                    u (torch.autograd.Variable): Residuals.
                    v (torch.autograd.Variable): Test Data.
                Returns:
                    float: Independence score.
        """

        x = u.numpy()
        y = v.numpy()

        n = x.shape[0]

        h = np.identity(n) - np.ones((n, n), dtype=float) / n

        k = HilbertSchmidtInformationCriterion.rbf_dot(x, HilbertSchmidtInformationCriterion.get_width(x, n))
        l = HilbertSchmidtInformationCriterion.rbf_dot(y, HilbertSchmidtInformationCriterion.get_width(y, n))

        k_c = np.dot(np.dot(h, k), h)
        l_c = np.dot(np.dot(h, l), h)

        test_stat = np.sum(k_c.T * l_c) / n

        return test_stat

    @staticmethod
    def predict_causal_relation(a: float, b: float) -> int:
        """Predict causal relation.

                Args:
                    a (float): Hsic score of causal relation from A to B.
                    b (flaot): Hsic score of causal relation from B to A.
                Returns:
                    int: Independence score.
        """
        if a <= b:
            return 0  # Causal relation form A to B
        else:
            return 1  # Causal relation form B to A
