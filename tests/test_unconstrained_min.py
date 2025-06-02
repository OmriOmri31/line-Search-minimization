import unittest
import numpy as np
import sys
import os

# Add src directory to Python path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.dirname(__file__))

from unconstrained_min import line_search_minimize
import utils
import examples


class TestUnconstrainedMin(unittest.TestCase):
    """
    Test class for unconstrained minimization algorithms.
    Tests both Gradient Descent and Newton's method on various objective functions.
    """

    def setUp(self):
        """Set up test parameters used across all test methods."""
        # Standard tolerances as specified in assignment
        self.obj_tol = 1e-12
        self.param_tol = 1e-8
        self.max_iter = 100
        self.max_iter_rosenbrock_gd = 10000  # Special case for Rosenbrock with GD

        # Standard starting point for most examples
        self.x0_standard = np.array([1.0, 1.0])
        # Special starting point for Rosenbrock
        self.x0_rosenbrock = np.array([-1.0, 2.0])

    def run_optimization_test(self, func, func_name, x0, xlim, ylim, special_max_iter=None):
        """
        Helper method to run optimization test for a given function.

        Parameters:
        - func: objective function to minimize
        - func_name: name of the function for display
        - x0: starting point
        - xlim, ylim: plot limits
        - special_max_iter: special max_iter for gradient descent (e.g., Rosenbrock)
        """
        print(f"\n{'=' * 60}")
        print(f"TESTING {func_name.upper()}")
        print(f"{'=' * 60}")

        # Run Gradient Descent
        max_iter_gd = special_max_iter if special_max_iter else self.max_iter
        final_x_gd, final_f_gd, success_gd, path_x_gd, path_f_gd = line_search_minimize(
            func, x0, self.obj_tol, self.param_tol, max_iter_gd, method='gd'
        )

        print(f"\nGRADIENT DESCENT FINAL RESULT:")
        print(f"Final x: {final_x_gd}")
        print(f"Final f(x): {final_f_gd:.12f}")
        print(f"Success: {success_gd}")
        print(f"Iterations: {len(path_f_gd)}")

        # Run Newton's Method
        final_x_nt, final_f_nt, success_nt, path_x_nt, path_f_nt = line_search_minimize(
            func, x0, self.obj_tol, self.param_tol, self.max_iter, method='newton'
        )

        print(f"\nNEWTON'S METHOD FINAL RESULT:")
        print(f"Final x: {final_x_nt}")
        print(f"Final f(x): {final_f_nt:.12f}")
        print(f"Success: {success_nt}")
        print(f"Iterations: {len(path_f_nt)}")

        # Create plots
        # 1. Contour plot with optimization paths
        fig1 = utils.plot_contours_with_path(
            func, xlim, ylim,
            f"Contour Lines and Optimization Paths - {func_name}",
            paths=[path_x_gd, path_x_nt],
            method_names=['Gradient Descent', "Newton's Method"]
        )

        # 2. Function values vs iterations
        iterations_gd = list(range(len(path_f_gd)))
        iterations_nt = list(range(len(path_f_nt)))

        fig2 = utils.plot_function_values(
            [iterations_gd, iterations_nt],
            [path_f_gd, path_f_nt],
            ['Gradient Descent', "Newton's Method"],
            f"Function Value vs Iteration - {func_name}"
        )

        # Show plots (in actual testing environment, you might want to save instead)
        import matplotlib.pyplot as plt
        plt.show()

        return (final_x_gd, final_f_gd, success_gd), (final_x_nt, final_f_nt, success_nt)

    def test_quadratic1(self):
        """Test optimization on quadratic function with circular contours."""
        self.run_optimization_test(
            examples.quadratic1,
            "Quadratic 1 (Circular Contours)",
            self.x0_standard,
            xlim=(-1.5, 1.5),
            ylim=(-1.5, 1.5)
        )

    def test_quadratic2(self):
        """Test optimization on quadratic function with axis-aligned elliptical contours."""
        self.run_optimization_test(
            examples.quadratic2,
            "Quadratic 2 (Axis-Aligned Ellipses)",
            self.x0_standard,
            xlim=(-1.2, 1.2),
            ylim=(-0.4, 0.4)
        )

    def test_quadratic3(self):
        """Test optimization on quadratic function with rotated elliptical contours."""
        self.run_optimization_test(
            examples.quadratic3,
            "Quadratic 3 (Rotated Ellipses)",
            self.x0_standard,
            xlim=(-1.2, 1.2),
            ylim=(-1.2, 1.2)
        )

    def test_rosenbrock(self):
        """Test optimization on the Rosenbrock function (non-convex, banana-shaped)."""
        self.run_optimization_test(
            examples.rosenbrock,
            "Rosenbrock Function",
            self.x0_rosenbrock,
            xlim=(-2, 2),
            ylim=(-1, 3),
            special_max_iter=self.max_iter_rosenbrock_gd
        )

    def test_linear(self):
        """Test optimization on linear function."""
        # Note: Linear functions don't have a minimum, so this test will likely reach max iterations
        self.run_optimization_test(
            examples.linear,
            "Linear Function",
            self.x0_standard,
            xlim=(-2, 2),
            ylim=(-2, 2)
        )

    def test_exponential_sum(self):
        """Test optimization on exponential sum function with triangle-like contours."""
        self.run_optimization_test(
            examples.exponential_sum,
            "Exponential Sum Function",
            self.x0_standard,
            xlim=(-2, 2),
            ylim=(-1, 1)
        )


if __name__ == '__main__':
    """
    Run all tests when script is executed directly.
    This will test both Gradient Descent and Newton's method on all example functions.
    """
    # Create test suite
    unittest.main(verbosity=2)