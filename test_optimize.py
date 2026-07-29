"""Self-checks for optimize_portfolio: constraints hold + RC-enforced path works."""
import numpy as np
import optimize_portfolio as m


def test_constraints_and_rc():
    # 3 low-vol, low-correlation assets -> an RC-compliant portfolio must exist
    R = np.array([0.10, 0.12, 0.08])
    cov = np.diag([0.02, 0.03, 0.015])          # ann vols ~14-17%, below RC=20%
    res = m.optimize(R, cov, enforce_rc=True)
    assert res is not None, "feasible RC case wrongly reported infeasible"
    w = res.x
    assert abs(w.sum() - 1.0) < 1e-6, "weights must sum to 1"
    assert (w >= -1e-9).all() and (w <= m.W_MAX + 1e-9).all(), "bounds violated"
    vol = np.sqrt(w @ cov @ w)
    assert vol <= m.RC + 1e-6, f"RC breached: {vol:.3f} > {m.RC}"


def test_infeasible_returns_none_when_enforced():
    # 3 super-volatile assets -> no allocation under RC=20%, but fallback still solves
    R = np.array([0.5, 0.5, 0.5])
    cov = np.diag([0.5, 0.6, 0.55])              # ann vols ~70-77%
    assert m.optimize(R, cov, enforce_rc=True) is None
    assert m.optimize(R, cov, enforce_rc=False) is not None   # fallback still solves


if __name__ == "__main__":
    test_constraints_and_rc()
    test_infeasible_returns_none_when_enforced()
    print("OK  optimizer self-checks pass")
