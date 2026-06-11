# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""QP / SQP subproblems for interaction-mesh retargeting (no cvxpy).

Default: **dense** ``min 0.5·dq'Pdq + q'·dq`` with box bounds via
``scipy.optimize.minimize`` (L-BFGS-B) or **KKT** for purely quadratic + box.

Optional: **OSQP** Python API (still no cvxpy) when ``use_osqp=True`` and
inequality rows are present.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse as sp
from scipy.optimize import Bounds, minimize

from hhtools.retarget.interaction_mesh.laplacian_geometry import calculate_laplacian_matrix


def laplacian_delta_q_dense(
    J_L: NDArray[np.floating],
    lap0_vec: NDArray[np.floating],
    target_lap_vec: NDArray[np.floating],
    w: float,
) -> NDArray[np.float32]:
    """Dense normal-equation solve ``min ||J_L @ dq - b||^2`` with ``b = target - lap0``."""
    jl = np.asarray(J_L, dtype=np.float64)
    b = (np.asarray(target_lap_vec) - np.asarray(lap0_vec)).reshape(-1)
    m, n = jl.shape
    if b.shape[0] != m:
        raise ValueError(f"J_L rows {m} != b len {b.shape[0]}")
    lhs = jl.T @ jl + (1e-8 * w) * np.eye(n, dtype=np.float64)
    rhs = jl.T @ b * float(w)
    dq = np.linalg.solve(lhs, rhs)
    return dq.astype(np.float32, copy=False)


def build_kron_laplacian_jacobian(
    L: NDArray[np.floating], J_V: NDArray[np.floating]
) -> NDArray[np.float64]:
    """``kron(L, I3) @ J_V`` in dense form (small V only — dev / tests)."""
    Lm = np.asarray(L, dtype=np.float64)
    jv = np.asarray(J_V, dtype=np.float64)
    v = Lm.shape[0]
    if jv.shape[0] != 3 * v:
        raise ValueError(f"J_V rows {jv.shape[0]} expected {3 * v}")
    kron = np.kron(Lm, np.eye(3))
    return (kron @ jv).astype(np.float64, copy=False)


def sparse_kron_laplacian_jacobian(
    L: NDArray[np.floating], J_V: NDArray[np.floating],
) -> sp.csr_matrix:
    """Sparse ``kron(L, I3) @ J_V`` matching the holosoma formulation."""
    Lm = sp.csr_matrix(np.asarray(L, dtype=np.float64))
    jv = sp.csr_matrix(np.asarray(J_V, dtype=np.float64))
    v = Lm.shape[0]
    if jv.shape[0] != 3 * v:
        raise ValueError(f"J_V rows {jv.shape[0]} expected {3 * v}")
    kron = sp.kron(Lm, sp.eye(3, format="csr"), format="csr")
    return kron @ jv


def laplacian_matrix_for_vertices(
    vertices: NDArray[np.floating], adj_list: list[list[int]],
) -> NDArray[np.float64]:
    """Dense Laplacian matrix ``L`` for umbrella weights."""
    return calculate_laplacian_matrix(vertices, adj_list, uniform_weight=True)


class OsqpUnreliableError(RuntimeError):
    """Raised by :func:`solve_qp_osqp` when OSQP cannot return a usable solution.

    The SQP outer loop must treat this as "drop this frame's solve and
    reuse ``q_prev``" — silently falling back to a box-only solve drops
    the inequality (non-penetration) rows that were the reason OSQP
    became infeasible in the first place, which then lets the cost
    gradient push the foot through terrain and produces the
    single-frame pose spikes observed on holosoma parkour clips.
    """


@dataclass(frozen=True)
class QpMatrices:
    """Dense QP: ``min 0.5 dq' P dq + q_vec' dq``."""

    P: NDArray[np.float64]
    q_vec: NDArray[np.float64]


def assemble_laplacian_qp(
    J_L: NDArray[np.floating],
    lap0_vec: NDArray[np.floating],
    target_lap_vec: NDArray[np.floating],
    *,
    laplacian_weight: float,
    dq_regularization: float = 1e-4,
) -> QpMatrices:
    """``P = w·J_L'J_L + reg·I``, ``q_vec = -w·J_L' b`` with ``b = target - lap0``."""
    jl = np.asarray(J_L, dtype=np.float64)
    b = (np.asarray(target_lap_vec) - np.asarray(lap0_vec)).reshape(-1)
    w = float(laplacian_weight)
    reg = float(dq_regularization)
    p = w * (jl.T @ jl) + reg * np.eye(jl.shape[1], dtype=np.float64)
    qv = (-w * (jl.T @ b)).astype(np.float64, copy=False)
    return QpMatrices(P=p, q_vec=qv)


def solve_qp_box_lbfgsb(qp: QpMatrices, dq_lb: NDArray, dq_ub: NDArray) -> NDArray[np.float32]:
    """Box-constrained convex QP via L-BFGS-B (fast C implementation in SciPy).

    The solver is configured with generous iteration limits and a relaxed
    tolerance so the SQP outer loop can always make progress.  Hitting the
    iteration cap is acceptable — the intermediate solution is still feasible
    (within box bounds) and directionally useful.
    """

    def fun(dq: NDArray[np.float64]) -> float:
        return float(0.5 * dq @ qp.P @ dq + qp.q_vec @ dq)

    def grad(dq: NDArray[np.float64]) -> NDArray[np.float64]:
        return qp.P @ dq + qp.q_vec

    n = qp.P.shape[0]
    x0 = np.clip(np.zeros(n), dq_lb, dq_ub)
    res = minimize(
        fun,
        x0,
        jac=grad,
        method="L-BFGS-B",
        bounds=Bounds(dq_lb, dq_ub),
        options={"maxiter": 5000, "ftol": 1e-7},
    )
    return res.x.astype(np.float32, copy=False)


def solve_qp_osqp(
    P: NDArray[np.floating],
    q_vec: NDArray[np.floating],
    A: NDArray[np.floating],
    l: NDArray[np.floating],
    u: NDArray[np.floating],
    *,
    warm_x: NDArray[np.floating] | None = None,
    verbose: bool = False,
) -> NDArray[np.float32]:
    """OSQP Python interface: ``l <= A x <= u``. ``P`` must be symmetric PSD.

    Tolerances and iteration cap are tuned for the SQP outer loop:
    ``eps_abs/eps_rel = 1e-4`` is enough accuracy for one inner step
    (the next iter relinearises anyway) while being orders of magnitude
    less likely to time out on KKT-stiff parkour-contact frames where
    the previous ``1e-6`` setting was hitting OSQP's default 4000-iter
    cap and falling out as ``MAX_ITER_REACHED``.

    Status handling: ``SOLVED`` (1) and ``SOLVED_INACCURATE`` (2) are
    returned as-is; everything else (``MAX_ITER_REACHED``,
    ``PRIMAL_INFEASIBLE``, ``DUAL_INFEASIBLE``, ``NON_CVX``, …) raises
    :class:`OsqpUnreliableError` so the SQP outer loop can roll the
    affected frame back to ``q_prev`` rather than committing a
    half-converged step.
    """
    import osqp
    import scipy.sparse as sps

    Pm = sps.csc_matrix(np.asarray(P, dtype=np.float64))
    Am = sps.csc_matrix(np.asarray(A, dtype=np.float64))
    qv = np.asarray(q_vec, dtype=np.float64).reshape(-1)
    ll = np.asarray(l, dtype=np.float64).reshape(-1)
    uu = np.asarray(u, dtype=np.float64).reshape(-1)
    m = Am.shape[0]
    n = Pm.shape[0]
    if Am.shape[1] != n or ll.shape[0] != m or uu.shape[0] != m:
        raise ValueError("OSQP dimension mismatch")

    prob = osqp.OSQP()
    prob.setup(
        Pm,
        qv,
        Am,
        ll,
        uu,
        verbose=verbose,
        # ``polish=True`` runs OSQP's post-ADMM Newton polish step on
        # the detected active set, giving solutions accurate to
        # ``eps_polish`` (default 1e-7) instead of the ADMM-iterate
        # accuracy of ~1e-3.  In an SQP outer loop the ~1e-3 noise
        # is otherwise re-injected into the next linearisation and
        # propagates as multi-degree per-frame ``Δq`` jitter that
        # ``smooth_weight`` cannot remove because the residual it
        # smooths is itself noisy.  A polished solution costs ~1.2×
        # the unpolished one and removes the jitter at its source.
        polish=True,
        eps_abs=1e-4,
        eps_rel=1e-4,
        max_iter=2000,
        warm_start=True,
    )
    if warm_x is not None:
        wx = np.asarray(warm_x, dtype=np.float64).reshape(-1)
        if wx.shape[0] == n:
            prob.warm_start(x=wx)
    # OSQP still prints polish status ("Polishing not needed - …") to stdout/stderr
    # even when verbose=False; suppress unless the caller asked for solver output.
    if verbose:
        r = prob.solve()
    else:
        with open(os.devnull, "w", encoding="utf-8") as dev:
            with contextlib.redirect_stdout(dev), contextlib.redirect_stderr(dev):
                r = prob.solve()
    if r.info.status_val not in (1, 2):  # solved or solved inaccurate
        raise OsqpUnreliableError(f"OSQP failed status={r.info.status}")
    return r.x.astype(np.float32, copy=False)


__all__ = [
    "OsqpUnreliableError",
    "QpMatrices",
    "assemble_laplacian_qp",
    "build_kron_laplacian_jacobian",
    "laplacian_delta_q_dense",
    "laplacian_matrix_for_vertices",
    "solve_qp_box_lbfgsb",
    "solve_qp_osqp",
    "sparse_kron_laplacian_jacobian",
]
