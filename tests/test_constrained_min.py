import unittest
import numpy as np

from src.constrained_min import interior_pt
from examples import qp_problem, lp_problem

TOL_QP = 1e-5
TOL_LP = 1e-3


class ConstrainedMinTests(unittest.TestCase):
    """Demonstrate the QP and LP examples solve correctly."""

    def test_qp(self):
        f, g_list, A, b, x0, sol = qp_problem()
        x_star, _ = interior_pt(f, g_list, A, b, x0)
        self.assertTrue(
            np.allclose(x_star, sol, atol=TOL_QP),
            msg=f"QP failed: expected {sol}, got {x_star}"
        )

    def test_lp(self):
        f, g_list, A, b, x0, sol = lp_problem()
        x_star, _ = interior_pt(f, g_list, A, b, x0)
        self.assertTrue(
            np.allclose(x_star, sol, atol=TOL_LP),
            msg=f"LP failed: expected {sol}, got {x_star}"
        )

if __name__ == "__main__":
    unittest.main(verbosity=2)
