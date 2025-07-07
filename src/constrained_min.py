from __future__ import annotations
from typing import Callable, List, Tuple, Dict, Optional

import numpy as np
from numpy.linalg import solve, norm, pinv

Array = np.ndarray


def interior_pt(
    func: Callable[[Array], float],
    ineq_constraints: List[Callable[[Array], float]],
    eq_constraints_mat: Optional[Array],
    eq_constraints_rhs: Optional[Array],
    x0: Array,
    *,
    t0: float = 1.0,
    mu: float = 10.0,
    tol_outer: float = 1e-8,
    tol_newton: float = 1e-10,
    max_outer: int = 50,
    max_newton: int = 50,
) -> Tuple[Array, Dict[str, list]]:
    """
    Interior-point algorithm with log barrier.

    Returns
    -------
    x_star : np.ndarray
        Final point.
    hist   : dict
        Keys:
            'centers' – list of iterates (one per outer iteration)
            'f'       – objective value at those iterates
    """
    # ------------------------------------------------------------------
    #  Sanity -- check that derivatives are attached
    # ------------------------------------------------------------------
    _check_callable_has_derivatives(func, "objective function")
    for gi in ineq_constraints:
        _check_callable_has_derivatives(gi, "constraint")

    #  Set-up

    m = len(ineq_constraints)
    n = x0.size

    if eq_constraints_mat is None:
        A = np.empty((0, n))
        b = np.empty((0,))
    else:
        A = np.asarray(eq_constraints_mat, dtype=float)
        b = np.asarray(eq_constraints_rhs, dtype=float)
        assert A.shape[0] == b.size, "A and b incompatible shapes"
        assert A.shape[1] == n, "A has wrong #columns"

    for g in ineq_constraints:
        assert g(x0) < 0, "x0 not strictly feasible (g_i>=0)"
    if A.size:
        assert norm(A @ x0 - b, np.inf) < 1e-10, "x0 violates equalities"

    hist = {"centers": [x0.copy()], "f": [func(x0)]}

    x = x0.copy()
    t = float(t0)

    for _ in range(max_outer):
        if m / t < tol_outer:
            break

        for _ in range(max_newton):
            grad_phi, hess_phi = _central_grad_hess(func, ineq_constraints, x, t)
            dx, _ = _newton_step(hess_phi, grad_phi, A)  # equality RHS = 0

            # Newton decrement  λ²/2  :=  Δxᵀ H Δx / 2
            if (dx @ hess_phi @ dx) / 2 <= tol_newton:
                break

            # back-tracking that keeps strict feasibility
            alpha = _backtracking_line_search(
                x, dx, func, ineq_constraints, t, grad_phi
            )
            x = x + alpha * dx

        # bookkeeping + update barrier parameter
        hist["centers"].append(x.copy())
        hist["f"].append(func(x))
        t *= mu

    return x, hist


# ----------------------------------------------------------------------
#  Helpers
# ----------------------------------------------------------------------
def _check_callable_has_derivatives(f, tag: str) -> None:
    """Ensure `f` has gradiant and hessian – clearer failure than AttributeError."""
    if not (hasattr(f, "grad") and hasattr(f, "hess")):
        raise AttributeError(f"{tag} must expose .grad and .hess attributes.")


def _central_grad_hess(func, g_list, x: Array, t: float) -> Tuple[Array, Array]:
    """Gradient & Hessian"""
    grad = t * func.grad(x)
    hess = t * func.hess(x)

    for g in g_list:
        g_val = g(x)                     # g_val < 0   (strict interior)
        g_grad = g.grad(x)
        g_hess = g.hess(x)

        grad += g_grad / (-g_val)
        hess += (np.outer(g_grad, g_grad) / g_val**2) - (g_hess / g_val)

    return grad, hess


def _newton_step(
    hess_phi: Array,
    grad_phi: Array,
    A: Array,
    reg: float = 1e-8,
) -> Tuple[Array, Array]:
    """Return (Δx, λ) solving the KKT system"""
    n = grad_phi.size
    m_eq = A.shape[0]
    H = hess_phi + reg * np.eye(n)

    if m_eq == 0:
        try:
            dx = -solve(H, grad_phi)
        except np.linalg.LinAlgError:
            dx = -(pinv(H) @ grad_phi)
        return dx, np.zeros(0)

    # build full KKT matrix
    KKT = np.block([[H, A.T],
                    [A, np.zeros((m_eq, m_eq))]])
    rhs = -np.concatenate([grad_phi, np.zeros(m_eq)])

    try:
        sol = solve(KKT, rhs)
    except np.linalg.LinAlgError:
        sol = pinv(KKT) @ rhs

    dx = sol[:n]
    lam = sol[n:]
    return dx, lam


def _backtracking_line_search(
    x: Array,
    dx: Array,
    func,
    g_list,
    t: float,
    grad_phi: Array,
    alpha0: float = 1.0,
    beta: float = 0.5,
    c: float = 1e-4,
) -> float:
    """Back-tracking that (1) enforces strict feasibility, then (2) Armijo."""
    alpha_max = alpha0
    for g in g_list:
        g_val = g(x)
        slope = g.grad(x) @ dx
        if slope > 0:                    # g increases along dx
            alpha_max = min(alpha_max, -g_val / slope)
    alpha = 0.99 * alpha_max             # shrink a tad to stay strict


    def phi(z):
        return t * func(z) - sum(np.log(-g(z)) for g in g_list)

    phi_x = phi(x)
    while alpha > 1e-16:
        x_trial = x + alpha * dx
        if all(g(x_trial) < 0 for g in g_list):
            if phi(x_trial) <= phi_x + c * alpha * (grad_phi @ dx):
                return alpha
        alpha *= beta                   # shrink and retry

    return alpha
