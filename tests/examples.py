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