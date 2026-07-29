"""
optimize_portfolio.py  —  Model 3 two-stage optimizer (Steps 2 & 3).

Reads the cached artifacts from build_universe.py and runs the two-stage
personalised portfolio optimisation on REAL CSE data, then prints the
formatted recommendation report.

    Stage 1  sector screening   S_i = R_sector,i * (1 + alpha * T_i)   -> top K
    Stage 2  stock calibration  R_ij^adj = R_ij * (1 + alpha * beta_ij * T_i)
    Stage 3  SLSQP utility max   max_W  W.R_adj - lambda * W'SW
             s.t. sum w = 1,  0 <= w <= w_max,  sqrt(W'SW) <= RC

Run:  python optimize_portfolio.py   (after build_universe.py has written cache/)
"""
import os, pickle
import numpy as np, pandas as pd
from scipy.optimize import minimize
try:
    from sklearn.covariance import LedoitWolf
    HAVE_LW = True
except ImportError:
    HAVE_LW = False

HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")

# ---- investor & macro inputs (from Model 1 / spec Step 2) ----------------- #
F     = 1_000_000     # available capital (LKR)
BRI   = 0.70          # behavioural risk index  (0..1)
ALPHA = 0.50          # trend willingness       (0.2 low / 0.5 med / 0.8 high)
RC    = 0.20          # risk capacity: max annualised volatility (20%)
W_MAX = 0.35          # single-stock concentration cap
TOP_K = 3             # sectors to select in Stage 1
LAMBDA = 1.0 - BRI    # risk-aversion penalty

# sector trend scores T_i (macro view, -1..+1), keyed to GICS sector names
TREND_SCORES = {
    "Banks": 0.80,
    "Telecommunication Services": 0.60,
    "Retailing": 0.20,
    "Transportation": -0.40,
    "Food, Beverage & Tobacco": 0.10,
    # any sector not listed defaults to 0.0 (neutral)
}


def load_cache():
    uni  = pd.read_csv(os.path.join(CACHE, "stock_universe.csv"))
    with open(os.path.join(CACHE, "returns.pkl"), "rb") as f:
        rets = pickle.load(f)
    sect = pd.read_csv(os.path.join(CACHE, "sector_returns.csv"))
    return uni, rets, sect


# --------------------------------------------------------------------------- #
# Stage 1 — sector screening (only sectors that hold investable stocks)
# --------------------------------------------------------------------------- #
def stage1(uni, sect):
    investable = set(uni["Sector"].unique())
    s = sect[sect["Sector"].isin(investable)].copy()
    s["Trend_Score"]    = s["Sector"].map(TREND_SCORES).fillna(0.0)
    s["Adjusted_Score"] = s["Expected_Return"] * (1 + ALPHA * s["Trend_Score"])
    top = s.sort_values("Adjusted_Score", ascending=False).head(TOP_K).reset_index(drop=True)
    return top


# --------------------------------------------------------------------------- #
# Stage 2 — beta-adjusted stock returns for the selected sectors
# --------------------------------------------------------------------------- #
def stage2(uni, top):
    q = uni[uni["Sector"].isin(top["Sector"])].copy()
    q = q.merge(top[["Sector", "Trend_Score"]], on="Sector", how="left")
    q["R_adj"] = q["Expected_Return"] * (1 + ALPHA * q["Beta"] * q["Trend_Score"])
    return q.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Stage 3 — SLSQP optimisation with robustness fixes
# --------------------------------------------------------------------------- #
def covariance(rets, tickers):
    r = rets[tickers].dropna()
    if HAVE_LW and len(r) > len(tickers):
        return LedoitWolf().fit(r.values).covariance_ * 252   # shrunk, well-conditioned
    return np.cov(r.values, rowvar=False) * 252


def optimize(R_adj, cov, enforce_rc=True):
    n = len(R_adj)

    def neg_utility(w):
        return -(w @ R_adj - LAMBDA * (w @ cov @ w))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]      # full investment
    if enforce_rc:
        cons.append({"type": "ineq",                              # smooth: RC^2 - variance >= 0
                     "fun": lambda w: RC**2 - w @ cov @ w})
    bounds = [(0.0, W_MAX)] * n

    # multiple starts (SLSQP is local) -> keep the best feasible optimum
    best = None
    starts = [np.ones(n) / n] + [_dirichlet_like(n, k) for k in range(6)]
    for x0 in starts:
        res = minimize(neg_utility, x0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
        if res.success and (best is None or res.fun < best.fun):
            best = res
    return best


def _dirichlet_like(n, seed):
    # deterministic spread of starting points without RNG (Math.random-free)
    v = np.array([((seed * 7 + i * 13) % 11) + 1 for i in range(n)], float)
    return v / v.sum()


# --------------------------------------------------------------------------- #
# Step 3 — formatted report
# --------------------------------------------------------------------------- #
def report(top, q, weights, cov, rc_enforced=True):
    port_ret = weights @ q["R_adj"].values
    port_vol = float(np.sqrt(weights @ cov @ weights))
    if not rc_enforced:
        print("\n*** NOTE: no allocation met RC = {:.0%} (CSE single-stock vols are 26-72%)."
              "\n    Showing the best utility portfolio within concentration limits;"
              "\n    the RC check below is expected to read FAIL. Raise RC or widen the"
              "\n    universe (extend ticker_sector.csv) for an RC-compliant result. ***"
              .format(RC))

    print("\n" + "=" * 64)
    print("STAGE 1 — SECTOR SCREENING (top {} of investable sectors)".format(TOP_K))
    print("=" * 64)
    print(top[["Sector", "Expected_Return", "Trend_Score", "Adjusted_Score"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 64)
    print("STAGE 2 — QUALIFIED STOCKS (beta & trend adjusted)")
    print("=" * 64)
    print(q[["Ticker", "Sector", "Beta", "Expected_Return", "R_adj"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    alloc = q[["Ticker", "Sector"]].copy()
    alloc["Weight_%"]       = (weights * 100).round(2)
    alloc["Funds_LKR"]      = (weights * F).round(0)
    alloc = alloc[alloc["Weight_%"] > 0.01].sort_values("Weight_%", ascending=False)

    print("\n" + "=" * 64)
    print("STAGE 3 — OPTIMAL PORTFOLIO ALLOCATION")
    print("=" * 64)
    print(alloc.to_string(index=False))

    print("\n" + "=" * 64)
    print("PORTFOLIO KPIs")
    print("=" * 64)
    print(f"  Total Capital Invested        : LKR {F:,.2f}")
    print(f"  Expected Annualised Return    : {port_ret*100:.2f}%")
    print(f"  Expected Annualised Volatility: {port_vol*100:.2f}%")
    print(f"  Risk Capacity Limit (RC)      : {RC*100:.2f}%")
    print(f"  Risk Capacity Check           : {'PASS' if port_vol <= RC + 1e-6 else 'FAIL'}")
    print(f"  Applied Risk Penalty (lambda) : {LAMBDA:.2f}  (BRI={BRI})")
    print(f"  Covariance estimator          : {'Ledoit-Wolf shrinkage' if HAVE_LW else 'sample'}")


def main():
    uni, rets, sect = load_cache()
    top = stage1(uni, sect)
    q   = stage2(uni, top)

    need = int(np.ceil(1.0 / W_MAX))              # 35% cap needs >=3 stocks to reach sum=1
    if len(q) < need:
        print(f"INFEASIBLE: only {len(q)} stock(s) in the selected sectors, but the "
              f"{W_MAX:.0%} concentration cap needs at least {need}. "
              f"Extend ticker_sector.csv so more stocks qualify.")
        return

    cov = covariance(rets, q["Ticker"].tolist())
    res = optimize(q["R_adj"].values, cov, enforce_rc=True)
    rc_enforced = res is not None
    if res is None:
        res = optimize(q["R_adj"].values, cov, enforce_rc=False)   # fallback: best utility
    if res is None:
        print("INFEASIBLE: optimiser failed even without the RC constraint.")
        return

    report(top, q, res.x, cov, rc_enforced=rc_enforced)


if __name__ == "__main__":
    main()
