import numpy as np


def line_search_minimize(f, x0, obj_tol=1e-12, param_tol=1e-8, max_iter=100, method='gd'):
    """
    Line search minimization with Gradient Descent or Newton's method.

    Parameters:
    - f: objective function that returns (value, gradient, hessian)
    - x0: starting point (numpy array)
    - obj_tol: tolerance for objective function change
    - param_tol: tolerance for parameter change
    - max_iter: maximum number of iterations
    - method: 'gd' for gradient descent, 'newton' for Newton's method

    Returns:
    - final_x: final location
    - final_f: final objective value
    - success: boolean flag for success/failure
    - path_x: list of x values at each iteration
    - path_f: list of function values at each iteration
    """

    # Initialize current point and storage for path
    x = np.array(x0, dtype=float)
    path_x = [x.copy()]
    path_f = []

    # Wolfe condition parameters
    c1 = 0.001  # Armijo condition parameter
    backtrack_factor = 0.5  # backtracking constant

    print(f"Starting {method.upper()} optimization from x0 = {x}")

    for i in range(max_iter):
        # Evaluate function, gradient, and hessian if needed
        need_hessian = (method == 'newton') # We will only need to claculate the hassian if we use newton's
        f_val, grad, hess = f(x, need_hessian)

        # Store current function value
        path_f.append(f_val)

        # current iteration
        print(f"Iteration {i}: x = {x}, f(x) = {f_val}")

        # Check gradient convergence (Newton decrement for Newton method)
        if method == 'newton' and hess is not None:
            # Newton decrement: sqrt(gradiant^T * Hess^(-1) * grad)
            try:
                newton_decrement = np.sqrt(grad.T @ np.linalg.solve(hess, grad))
                if newton_decrement < np.sqrt(2 * obj_tol):
                    print(f"Converged: Newton decrement {newton_decrement} < {np.sqrt(2 * obj_tol)}")
                    return x, f_val, True, path_x, path_f
            except np.linalg.LinAlgError:
                print("Hessian is singular, switching to gradient descent step")
                method_this_iter = 'gd'
            else:
                method_this_iter = 'newton'
        else:
            method_this_iter = 'gd'
            # Check gradient norm for gradient descent
            if np.linalg.norm(grad) < np.sqrt(obj_tol):
                print(f"Converged: gradient norm {np.linalg.norm(grad)} < {np.sqrt(obj_tol)}")
                return x, f_val, True, path_x, path_f

        # Compute search direction
        if method_this_iter == 'newton':
            try:
                direction = -np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                # Gradient descent if Hessian is singular
                direction = -grad
        else:
            # Gradient descent direction: -grad
            direction = -grad

        # Line search with backtracking Wolfe conditions
        alpha = 1.0  # Initial step size , we will be shriniking it by multiplying it by backstrack_factor  = 0.5

        # Backtracking line search
        while True:
            x_new = x + alpha * direction
            f_new, _, _ = f(x_new, False)

            # Armijo condition: f(x + alpha* d) <= f(x) + c1*alpha*grad^T*d
            Armijo_condition = f_new <= f_val + c1 * alpha * np.dot(grad, direction)

            if Armijo_condition or alpha < 1e-16: # Accept step if Armijo is satisfied, or alpha is small to continue.
                break

            alpha *= backtrack_factor

        # Update x
        x_prev = x.copy()
        x = x + alpha * direction
        path_x.append(x.copy())

        # Check parameter tolerance
        param_change = np.linalg.norm(x - x_prev)
        if param_change < param_tol:
            print(f"Converged: parameter change {param_change} < {param_tol}")
            f_val, _, _ = f(x, False)
            path_f.append(f_val)
            print(f"Final iteration {i + 1}: x = {x}, f(x) = {f_val}")
            return x, f_val, True, path_x, path_f

        # Check objective function change
        if i > 0:
            obj_change = abs(f_val - path_f[-2])  # Compare with previous iteration
            if obj_change < obj_tol:
                print(f"Converged: objective change {obj_change} < {obj_tol}")
                f_val, _, _ = f(x, False)
                path_f.append(f_val)
                print(f"Final iteration {i + 1}: x = {x}, f(x) = {f_val}")
                return x, f_val, True, path_x, path_f

    # Maximum iterations reached
    f_val, _, _ = f(x, False)
    path_f.append(f_val)
    print(f"Maximum iterations reached. Final: x = {x}, f(x) = {f_val}")
    return x, f_val, False, path_x, path_f