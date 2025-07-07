import numpy as np


def quadratic1(x, hessian_needed=False):
    """
    Quadratic function f(x) = x^T * Q * x where Q = [[1, 0], [0, 1]] (identity matrix).
    Contour lines are circles.

    Parameters:
    - x: input vector (2D)
    - hessian_needed: whether to compute and return Hessian

    Returns:
    - f: function value
    - g: gradient vector
    - h: Hessian matrix (if needed)
    """
    x = np.array(x, dtype=float)
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])  # Identity matrix

    # Function value: f = x^T * Q * x
    f = x.T @ Q @ x

    # Gradient: grad = 2 * Q * x (since d/dx[x^T*Q*x] = (Q + Q^T)*x = 2*Q*x for symmetric Q)
    g = 2 * Q @ x

    # Hessian: H = 2 * Q (constant for quadratic functions)
    h = 2 * Q if hessian_needed else None

    return f, g, h


def quadratic2(x, hessian_needed=False):
    """
    Quadratic function f(x) = x^T * Q * x where Q = [[1, 0], [0, 100]].
    Contour lines are axis-aligned ellipses.

    Parameters:
    - x: input vector (2D)
    - hessian_needed: whether to compute and return Hessian

    Returns:
    - f: function value
    - g: gradient vector
    - h: Hessian matrix (if needed)
    """
    x = np.array(x, dtype=float)
    Q = np.array([[1.0, 0.0], [0.0, 100.0]])  # Diagonal matrix with different eigenvalues

    # Function value: f = x^T * Q * x
    f = x.T @ Q @ x

    # Gradient: grad = 2 * Q * x
    g = 2 * Q @ x

    # Hessian: H = 2 * Q
    h = 2 * Q if hessian_needed else None

    return f, g, h


def quadratic3(x, hessian_needed=False):
    """
    Quadratic function f(x) = x^T * Q * x where Q is a rotated ellipse matrix.
    Q = R^T * D * R where R is rotation matrix and D is diagonal.
    Contour lines are rotated ellipses.

    Parameters:
    - x: input vector (2D)
    - hessian_needed: whether to compute and return Hessian

    Returns:
    - f: function value
    - g: gradient vector
    - h: Hessian matrix (if needed)
    """
    x = np.array(x, dtype=float)

    # Rotation matrix R (30 degrees rotation)
    # R = [[sqrt(3)/2, -0.5], [0.5, sqrt(3)/2]]
    sqrt3_2 = np.sqrt(3) / 2
    R = np.array([[sqrt3_2, -0.5], [0.5, sqrt3_2]])

    # Diagonal matrix D with [100, 1]on the dioagon
    D = np.array([[100.0, 0.0], [0.0, 1.0]])

    # Final Q matrix: Q =  R^T * D * R
    Q = R.T @ D @ R

    # Function value: f = x^T * Q * x
    f = x.T @ Q @ x

    # Gradient: grad = 2 * Q * x
    g = 2 * Q @ x

    # Hessian: H = 2 * Q
    h = 2 * Q if hessian_needed else None

    return f, g, h


def rosenbrock(x, hessian_needed=False):
    """
    Rosenbrock function: f(x) = 100*(x2 - x1^2)^2 + (1 - x1)^2
    Famous non-convex optimization benchmark with banana-shaped contours.

    Parameters:
    - x: input vector (2D)
    - hessian_needed: whether to compute and return Hessian

    Returns:
    - f: function value
    - g: gradient vector
    - h: Hessian matrix (if needed)
    """
    x = np.array(x, dtype=float)
    x1, x2 = x[0], x[1]

    # Function value: f = 100*(x2 - x1^2)^2 + (1 - x1)^2
    f = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2

    # Gradient computation:
    # df/dx1 = 100 * 2 * (x2 - x1^2) * (-2*x1) + 2 * (1 - x1) * (-1)
    #        = -400 * x1 * (x2 - x1^2) - 2 * (1 - x1)
    # df/dx2 = 100 * 2 * (x2 - x1^2) * 1 = 200 * (x2 - x1^2)
    g = np.array([
        -400 * x1 * (x2 - x1 ** 2) - 2 * (1 - x1),
        200 * (x2 - x1 ** 2)
    ])

    # Hessian computation (if needed):
    if hessian_needed:
        # d2f/dx1^2 = -400 * (x2 - x1^2) - 400 * x1 * (-2*x1) + 2
        #           = -400 * (x2 - x1^2) + 800 * x1^2 + 2
        h11 = -400 * (x2 - x1 ** 2) + 800 * x1 ** 2 + 2

        # d2f/dx1dx2 = d2f/dx2dx1 = -400 * x1
        h12 = h21 = -400 * x1

        # d2f/dx2^2 = 200
        h22 = 200

        h = np.array([[h11, h12], [h21, h22]])
    else:
        h = None

    return f, g, h


def linear(x, hessian_needed=False):
    """
    Linear function: f(x) = a^T * x where a is a chosen nonzero vector.
    Contour lines are straight lines.

    Parameters:
    - x: input vector (2D)
    - hessian_needed: whether to compute and return Hessian

    Returns:
    - f: function value
    - g: gradient vector
    - h: Hessian matrix (if needed)
    """
    x = np.array(x, dtype=float)
    a = np.array([2.0, -3.0])


    # Function value: f = a^T * x
    f = np.dot(a, x)

    # Gradient: grad = a (constant for linear functions)
    g = a.copy()

    # Hessian: H = 0 (zero matrix for linear functions)
    h = np.zeros((2, 2)) if hessian_needed else None

    return f, g, h


def exponential_sum(x, hessian_needed=False):
    """
    Exponential sum function: f(x1, x2) = exp(x1 + 3*x2 - 0.1) + exp(x1 - 3*x2 - 0.1) + exp(-x1 - 0.1)
    Contour lines look like smoothed corner triangles.
    From Boyd's book, p. 470, example 9.20.

    Parameters:
    - x: input vector (2D)
    - hessian_needed: whether to compute and return Hessian

    Returns:
    - f: function value
    - g: gradient vector
    - h: Hessian matrix (if needed)
    """
    x = np.array(x, dtype=float)
    x1, x2 = x[0], x[1]

    # Individual exponential terms
    exp1 = np.exp(x1 + 3 * x2 - 0.1)
    exp2 = np.exp(x1 - 3 * x2 - 0.1)
    exp3 = np.exp(-x1 - 0.1)

    # Function value: sum of exponentials
    f = exp1 + exp2 + exp3

    # Gradient computation:
    # df/dx1 = exp1 * 1 + exp2 * 1 + exp3 * (-1)
    # df/dx2 = exp1 * 3 + exp2 * (-3) + exp3 * 0
    g = np.array([
        exp1 + exp2 - exp3,
        3 * exp1 - 3 * exp2
    ])

    # Hessian computation (if needed):
    if hessian_needed:
        # d2f/dx1^2 = exp1 + exp2 + exp3
        h11 = exp1 + exp2 + exp3

        # d2f/dx1dx2 = d2f/dx2dx1 = exp1 * 3 + exp2 * (-3)
        h12 = h21 = 3 * exp1 - 3 * exp2

        # d2f/dx2^2 = exp1 * 9 + exp2 * 9
        h22 = 9 * exp1 + 9 * exp2

        h = np.array([[h11, h12], [h21, h22]])
    else:
        h = None

    return f, g, h



#HW 2 ---------------------------------------------------------------------
def qp_problem():
    """Minimise  x₁² + x₂² + (x₃+1)²  """
    def f(x):
        x = np.asarray(x, dtype=float)
        if x.size == 3:
            x1, x2, x3 = x
        elif x.size == 2:
            x1, x2 = x
            x3 = 1.0 - x1 - x2
        else:
            raise ValueError("Wrong dimentions")
        return x1**2 + x2**2 + (x3 + 1.0)**2



    def grad(x):
        x = np.asarray(x, dtype=float)
        if x.size == 3:
            return np.array([2*x[0], 2*x[1], 2*(x[2] + 1)])
        elif x.size == 2:
            u, v = x
            w = 1.0 - u - v

            dfdu = 2*u - 2*(w + 1)
            dfdv = 2*v - 2*(w + 1)

            return np.array([dfdu, dfdv])

    def hess(x):
        x = np.asarray(x, dtype=float)
        if x.size == 3:
            return np.diag([2.0, 2.0, 2.0])             # constant :)
        elif x.size == 2:
            # 2×2 Hessian on the (u,v) plane
            return np.array([[4.0, 2.0],
                             [2.0, 4.0]])

    # attach derivatives
    f.grad, f.hess = grad, hess

    # constrains
    g_list: List = []
    for i in range(3):
        g_list.append(lambda x, i=i: -x[i])
        g_list[-1].grad = lambda x, i=i: -np.eye(3)[i]
        g_list[-1].hess = lambda x: np.zeros((3, 3))

    A = np.ones((1, 3))
    b = np.array([1.0])

    x0  = np.array([0.1, 0.2, 0.7])
    sol = np.array([0.5, 0.5, 0.0])

    return f, g_list, A, b, x0, sol



def lp_problem():
    """Maximise  x + y ."""
    def f(x): return -(x[0]+  x[1])      # minimise the negative – turns into LP
    f.grad = lambda x: np.array([-1.0, -1.0])
    f.hess = lambda x: np.zeros((2, 2))

    g_list: List = []
    g_list.append(lambda x: -x[1] - x[0] + 1.0)
    g_list[-1].grad = lambda x: np.array([-1.0, -1.0])
    g_list[-1].hess = lambda x: np.zeros((2, 2))
    # y ≤ 1 -> y - 1 ≤ 0
    g_list.append(lambda x: x[1] - 1.0)
    g_list[-1].grad = lambda x: np.array([0.0, 1.0])
    g_list[-1].hess = lambda x: np.zeros((2, 2))
    # x ≤ 2-> x - 2 ≤ 0
    g_list.append(lambda x: x[0] - 2.0)
    g_list[-1].grad = lambda x: np.array([1.0, 0.0])
    g_list[-1].hess = lambda x: np.zeros((2, 2))
    # y ≥ 0  -> -y ≤ 0
    g_list.append(lambda x: -x[1])
    g_list[-1].grad = lambda x: np.array([0.0, -1.0])
    g_list[-1].hess = lambda x: np.zeros((2, 2))

    x0 = np.array([0.5, 0.75])        # interior start
    sol = np.array([2.0, 1.0])
    return f, g_list, None, None, x0, sol

################################################################
####PLOTTING :) #####

if __name__ == "__main__":
    import os, sys
    import matplotlib.pyplot as plt
    from typing import List
    from src.constrained_min import interior_pt
    from src import utils            # same helper as HW-01
    np.set_printoptions(precision=8, suppress=True)
    os.makedirs("plots", exist_ok=True)

    def run_demo(name: str,
                 f, g_list: List,
                 A, b,
                 x0: np.ndarray,
                 xlims, ylims):
        x_star, hist = interior_pt(f, g_list, A, b, x0)

        def wrapper(z, need_hess=False):
            """utils.plot_contours_with_path expects f, grad, hess."""
            return f(z), f.grad(z), f.hess(z)

        fig_path = utils.plot_contours_with_path(
            wrapper,
            xlim=xlims, ylim=ylims,
            title=f"{name}: central path (outer iters)",
            paths=[hist["centers"]],
            method_names=["central path"],
            levels=25,
        )
        ax = fig_path.axes[0]
        ax.plot(*x_star[:2] if x_star.size == 2 else x_star[:2], "r*", ms=10, label="final")
        ax.legend()
        utils.save_plot(fig_path, f"plots/{name.lower().replace(' ', '_')}_path.png")

        fig_obj = utils.plot_function_values(
            [list(range(len(hist["f"])))],
            [hist["f"]],
            ["outer iterations"],
            f"{name}: objective vs outer iteration"
        )
        utils.save_plot(fig_obj, f"plots/{name.lower().replace(' ', '_')}_obj.png")

        feas_resid = (
            max([g(x_star) for g in g_list]) if g_list else 0.0,
            np.max(np.abs(A @ x_star - b)) if A is not None else 0.0
        )
        print(f"\n{name} – final candidate")
        print(f"  x* = {x_star}")
        print(f"  objective = {f(x_star):.12g}")
        print(f"  max inequality residual  = {feas_resid[0]:.3e}")
        print(f"  max equality residual    = {feas_resid[1]:.3e}")

    print("\n=== Quadratic programme on simplex ===")
    f_qp, g_qp, A_qp, b_qp, x0_qp, _ = qp_problem()
    run_demo("QP on simplex", f_qp, g_qp, A_qp, b_qp, x0_qp,
             xlims=(-0.1, 1.1), ylims=(-0.1, 1.1))

    print("\n=== Linear programme on polygon ===")
    f_lp, g_lp, A_lp, b_lp, x0_lp, _ = lp_problem()
    run_demo("LP on polygon", f_lp, g_lp, A_lp, b_lp, x0_lp,
             xlims=(-0.2, 2.2), ylims=(-0.2, 1.2))

    print("Done")
    plt.show()
