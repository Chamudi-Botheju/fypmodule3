"""
build_universe.py  —  Model 3 data pipeline (Phase 1).

Turns the 15 raw CSE files into two cached artifacts consumed by the optimizer:

    cache/stock_universe.csv   [Ticker, Sector, Expected_Return, Beta,
                                Public_Float_Pct, Avg_Turnover_LKR]
    cache/returns.pkl          daily total-return matrix  (Date x Ticker)
    cache/sector_returns.csv   annualised return per GICS sector (Stage 1 input)

Run once:  python build_universe.py
Re-reads the raw files only if cache is missing (delete cache/ to rebuild).

Corporate-action scope: DIVIDENDS (via CSE cum/ex prices) + SPLITS (via
proportion). ponytail: scrip dividends / rights / bonus deferred — they are
name-only joins (no ticker) and comparatively rare; add them when a ticker
key or a name->ticker table is available.
"""
import os, glob, zipfile, io, warnings, pickle
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)

PRICE_ZIP   = "33Daily Shares Price List -2021-2025.zip"
FLOAT_ZIP   = "17 Public Holding-Quarterly.zip"
DIV_FILE    = "01Dividends.xls"
SPLIT_FILE  = "05Sub Division (Share Splits).xls"
INDEX_FILE  = "07Market Indices - Daily.xls"
GICS_FILE   = "39GICS-Daily.xlsx"
SECTORMAP   = "ticker_sector.csv"        # pluggable ticker -> GICS sector map
MIN_FLOAT   = 0.15                        # spec: drop stocks with float < 15%
TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def norm_ticker(t):
    """AAF-N-0000 / AAF.N0000 / (AAF,N,0000)  ->  canonical  AAF.N0000."""
    t = str(t).strip().upper()
    if "." in t:
        return t
    parts = t.split("-")
    if len(parts) == 3:
        return f"{parts[0]}.{parts[1]}{parts[2]}"
    return t


def with_header(raw, token, max_scan=15):
    """Find the real header row (raw files carry title/blank rows on top)."""
    token = token.strip().upper()
    for i in range(min(max_scan, len(raw))):
        row = [str(x).strip().upper() for x in raw.iloc[i].tolist()]
        if token in row:
            out = raw.iloc[i + 1:].copy()
            out.columns = [str(x).strip() for x in raw.iloc[i].tolist()]
            return out.reset_index(drop=True)
    raise ValueError(f"header token {token!r} not found")


def col(df, *subs):
    """First column whose name contains all `subs` (case-insensitive)."""
    for c in df.columns:
        cu = str(c).upper()
        if all(s.upper() in cu for s in subs):
            return c
    raise KeyError(f"no column matching {subs} in {list(df.columns)}")


def read_any(name_or_buf, ext):
    """Read every sheet of an excel file (or a csv) as raw (header=None)."""
    if ext == "csv":
        return {"csv": pd.read_csv(name_or_buf, header=None, dtype=str)}
    return pd.read_excel(name_or_buf, sheet_name=None, header=None)


# --------------------------------------------------------------------------- #
# 1. daily prices  (per-year files, quarterly sheets, mixed xls/csv)
# --------------------------------------------------------------------------- #
def load_prices():
    print("[1] loading daily prices ...")
    rows = []
    with zipfile.ZipFile(PRICE_ZIP) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith((".xls", ".xlsx", ".csv"))]
        for m in sorted(members):
            ext = m.rsplit(".", 1)[1].lower()
            sheets = read_any(io.BytesIO(zf.read(m)), ext)
            for _, raw in sheets.items():
                try:
                    df = with_header(raw, "COMPANY ID")
                except ValueError:
                    continue
                cid, mt, st = col(df, "COMPANY", "ID"), col(df, "MAIN", "TYPE"), col(df, "SUB", "TYPE")
                dt, cp = col(df, "TRADING", "DATE"), col(df, "CLOSE")
                try:
                    tvr = col(df, "TURNOVER")
                except KeyError:
                    tvr = None
                sub = pd.DataFrame({
                    "Ticker": (df[cid].astype(str).str.strip() + "." +
                               df[mt].astype(str).str.strip() + df[st].astype(str).str.strip()),
                    "Date":   pd.to_datetime(df[dt], format="%d-%b-%y", errors="coerce"),
                    "Close":  pd.to_numeric(df[cp], errors="coerce"),
                    "Turnover": pd.to_numeric(df[tvr], errors="coerce") if tvr else np.nan,
                })
                rows.append(sub)
    px = pd.concat(rows, ignore_index=True).dropna(subset=["Date", "Close"])
    px = px[px["Close"] > 0]
    px["Ticker"] = px["Ticker"].map(norm_ticker)
    px = px.sort_values(["Ticker", "Date"]).drop_duplicates(["Ticker", "Date"], keep="last")
    print(f"    {px['Ticker'].nunique()} tickers, {len(px):,} price rows "
          f"({px['Date'].min().date()} .. {px['Date'].max().date()})")
    return px


# --------------------------------------------------------------------------- #
# 2. corporate actions  ->  (Ticker, Date, factor)   factor applied to PRIOR prices
# --------------------------------------------------------------------------- #
def load_actions():
    print("[2] loading corporate actions (dividends + splits) ...")
    acts = []

    # dividends: factor = ex_price / cum_price  (< 1) on the ex-date
    for _, raw in pd.read_excel(DIV_FILE, sheet_name=None, header=None).items():
        try:
            df = with_header(raw, "DATE OF EX")
        except ValueError:
            try:
                df = with_header(raw, "SECURITY")
            except ValueError:
                continue
        try:
            sec, exd = col(df, "SECURITY"), col(df, "DATE OF EX")
            cum, exp = col(df, "CUM", "PRICE"), col(df, "EX", "PRICE")
        except KeyError:
            continue
        d = pd.DataFrame({
            "Ticker": df[sec].map(norm_ticker),
            "Date":   pd.to_datetime(df[exd], format="%d-%b-%y", errors="coerce"),
            "cum":    pd.to_numeric(df[cum], errors="coerce"),
            "ex":     pd.to_numeric(df[exp], errors="coerce"),
        }).dropna(subset=["Date", "cum", "ex"])
        d = d[(d["cum"] > 0) & (d["ex"] > 0)]
        d["factor"] = (d["ex"] / d["cum"]).clip(upper=1.0)   # dividend never raises price
        acts.append(d[["Ticker", "Date", "factor"]])

    # splits / sub-divisions: factor = old / new  on the effective date
    for _, raw in pd.read_excel(SPLIT_FILE, sheet_name=None, header=None).items():
        try:
            df = with_header(raw, "COMPANY ID")
        except ValueError:
            continue
        try:
            cid, eff = col(df, "COMPANY", "ID"), col(df, "EFFECTIVE", "DATE")
            oldp, newp = col(df, "OLD", "PROPORTION"), col(df, "NEW", "PROPORTION")
        except KeyError:
            continue
        d = pd.DataFrame({
            "Ticker": df[cid].map(norm_ticker),
            "Date":   pd.to_datetime(df[eff], errors="coerce"),
            "old":    pd.to_numeric(df[oldp], errors="coerce"),
            "new":    pd.to_numeric(df[newp], errors="coerce"),
        }).dropna(subset=["Date", "old", "new"])
        d = d[(d["old"] > 0) & (d["new"] > 0)]
        d["factor"] = d["old"] / d["new"]
        acts.append(d[["Ticker", "Date", "factor"]])

    out = pd.concat(acts, ignore_index=True) if acts else pd.DataFrame(columns=["Ticker", "Date", "factor"])
    print(f"    {len(out):,} corporate actions across {out['Ticker'].nunique()} tickers")
    return out


# --------------------------------------------------------------------------- #
# 3. back-adjust prices  ->  total-return matrix (Date x Ticker)
# --------------------------------------------------------------------------- #
def adjusted_returns(px, acts):
    print("[3] back-adjusting prices -> total returns ...")
    px = px.copy()
    px["adj_factor"] = 1.0
    # CRSP-style: every price strictly BEFORE an action date is scaled by its factor
    for tk, grp in acts.groupby("Ticker"):
        mask = px["Ticker"] == tk
        if not mask.any():
            continue
        pt = px.loc[mask]
        cum = np.ones(len(pt))
        for _, a in grp.iterrows():
            cum = cum * np.where(pt["Date"].values < np.datetime64(a["Date"]), a["factor"], 1.0)
        px.loc[mask, "adj_factor"] = cum
    px["AdjClose"] = px["Close"] * px["adj_factor"]
    wide = px.pivot(index="Date", columns="Ticker", values="AdjClose").sort_index()
    rets = wide.pct_change()
    # drop stocks that barely trade (need a usable return history)
    good = rets.notna().sum() >= 0.30 * len(rets)
    rets = rets.loc[:, good]
    print(f"    return matrix: {rets.shape[0]} days x {rets.shape[1]} tickers")
    return rets


# --------------------------------------------------------------------------- #
# 4. betas vs ASPI (All Share Price Index)  — computed, NOT the 2009 file
# --------------------------------------------------------------------------- #
def market_betas(rets):
    print("[4] computing betas vs ASPI ...")
    raw = pd.read_excel(INDEX_FILE, sheet_name=0, header=None)
    # find the ASPI column (label 'All Share Price Index' sits under the date col block)
    hdr_row = next(i for i in range(10) if raw.iloc[i].astype(str).str.contains("All Share", case=False).any())
    aspi_col = next(j for j in range(raw.shape[1])
                    if "ALL SHARE" in str(raw.iloc[hdr_row, j]).upper())
    idx = pd.DataFrame({
        "Date":  pd.to_datetime(raw.iloc[hdr_row + 1:, 0], errors="coerce"),
        "ASPI":  pd.to_numeric(raw.iloc[hdr_row + 1:, aspi_col], errors="coerce"),
    }).dropna()
    idx = idx[~idx["Date"].duplicated(keep="last")]
    mkt = idx.set_index("Date")["ASPI"].sort_index().pct_change()
    mkt = mkt.reindex(rets.index)
    var_m = mkt.var()
    betas = {}
    for tk in rets.columns:
        pair = pd.concat([rets[tk], mkt], axis=1).dropna()
        betas[tk] = pair.iloc[:, 0].cov(pair.iloc[:, 1]) / var_m if len(pair) > 60 else np.nan
    b = pd.Series(betas).fillna(1.0).clip(-1, 3)      # sanity clip; missing -> market beta
    print(f"    betas for {b.notna().sum()} tickers (median {b.median():.2f})")
    return b


# --------------------------------------------------------------------------- #
# 5. public float (latest quarter)  +  6. sector map
# --------------------------------------------------------------------------- #
def load_float():
    print("[5] loading public float (latest quarter) ...")
    best = {}
    with zipfile.ZipFile(FLOAT_ZIP) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith((".xls", ".xlsx"))]
        for m in sorted(members):                      # later years overwrite earlier
            for _, raw in pd.read_excel(io.BytesIO(zf.read(m)), sheet_name=None, header=None).items():
                try:
                    df = with_header(raw, "SECURITY")
                except ValueError:
                    continue
                try:
                    sec, fl = col(df, "SECURITY"), col(df, "FLOAT")
                except KeyError:
                    continue
                pct = (df[fl].astype(str).str.replace("%", "", regex=False)
                       .pipe(pd.to_numeric, errors="coerce"))
                for tk, v in zip(df[sec].map(norm_ticker), pct):
                    if pd.notna(v):
                        best[tk] = v / 100.0
    s = pd.Series(best, name="Public_Float_Pct")
    print(f"    float for {len(s)} tickers")
    return s


def load_sector_map():
    path = os.path.join(HERE, SECTORMAP)
    if not os.path.exists(path):
        print(f"[6] WARNING: {SECTORMAP} not found -> writing SEED map (liquid large-caps).")
        print("    ACTION NEEDED: extend it with CSE's official GICS company classification.")
        _write_seed_sector_map(path)
    m = pd.read_csv(path)
    m["Ticker"] = m["Ticker"].map(norm_ticker)
    print(f"[6] sector map: {len(m)} tickers, {m['Sector'].nunique()} sectors")
    return m.set_index("Ticker")["Sector"]


def _write_seed_sector_map(path):
    seed = {
        # Banks
        "COMB.N0000": "Banks", "HNB.N0000": "Banks", "SAMP.N0000": "Banks",
        "NDB.N0000": "Banks", "DFCC.N0000": "Banks", "SEYB.N0000": "Banks",
        "NTB.N0000": "Banks", "PABC.N0000": "Banks", "UBC.N0000": "Banks",
        # Telecommunication
        "DIAL.N0000": "Telecommunication Services", "SLTL.N0000": "Telecommunication Services",
        # Food, Beverage & Tobacco
        "NEST.N0000": "Food, Beverage & Tobacco", "CARG.N0000": "Food, Beverage & Tobacco",
        "LION.N0000": "Food, Beverage & Tobacco", "DIST.N0000": "Food, Beverage & Tobacco",
        "CTC.N0000": "Food, Beverage & Tobacco", "MELS.N0000": "Food, Beverage & Tobacco",
        # Capital Goods / Diversified
        "JKH.N0000": "Capital Goods", "HHL.N0000": "Capital Goods", "SPEN.N0000": "Capital Goods",
        "HAYL.N0000": "Capital Goods", "CARS.N0000": "Capital Goods", "RICH.N0000": "Capital Goods",
        "AEL.N0000": "Capital Goods",
        # Materials
        "TKYO.N0000": "Materials", "RCL.N0000": "Materials", "ACL.N0000": "Materials",
        # Consumer Durables & Apparel / Retail
        "MGT.N0000": "Consumer Durables & Apparel", "ODEL.N0000": "Retailing",
        # Diversified Financials / Insurance
        "LOLC.N0000": "Diversified Financials", "CFIN.N0000": "Diversified Financials",
        "AAIC.N0000": "Insurance", "CINS.N0000": "Insurance",
        # Transportation
        "EXPO.N0000": "Transportation", "LOLC.N0000": "Diversified Financials",
    }
    pd.DataFrame({"Ticker": list(seed), "Sector": list(seed.values())}).to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# 7. sector returns for Stage-1 screening (from GICS sector index levels)
# --------------------------------------------------------------------------- #
def sector_returns():
    print("[7] sector annualised returns (GICS index) ...")
    g = pd.read_excel(GICS_FILE, sheet_name="Sectors Index")
    g = g.rename(columns={g.columns[0]: "Date"})
    g["Date"] = pd.to_datetime(g["Date"], errors="coerce")
    g = g.dropna(subset=["Date"]).set_index("Date").sort_index().apply(pd.to_numeric, errors="coerce")
    daily = g.pct_change()
    ann = daily.mean() * TRADING_DAYS
    out = ann.rename("Expected_Return").rename_axis("Sector").reset_index()
    out.to_csv(os.path.join(CACHE, "sector_returns.csv"), index=False)
    print(f"    {len(out)} sectors -> cache/sector_returns.csv")
    return out


# --------------------------------------------------------------------------- #
# assemble
# --------------------------------------------------------------------------- #
def main():
    px    = load_prices()
    acts  = load_actions()
    rets  = adjusted_returns(px, acts)
    betas = market_betas(rets)
    flt   = load_float()
    smap  = load_sector_map()
    sector_returns()

    exp_ret = rets.mean() * TRADING_DAYS                       # spec: mean daily x 252
    turnover = px.groupby("Ticker")["Turnover"].mean()

    uni = pd.DataFrame({"Ticker": rets.columns})
    uni["Sector"]           = uni["Ticker"].map(smap)
    uni["Expected_Return"]  = uni["Ticker"].map(exp_ret)
    uni["Beta"]             = uni["Ticker"].map(betas)
    uni["Public_Float_Pct"] = uni["Ticker"].map(flt)
    uni["Avg_Turnover_LKR"] = uni["Ticker"].map(turnover)

    uni["Beta"] = uni["Beta"].fillna(1.0)
    before = len(uni)
    uni = uni[uni["Public_Float_Pct"].fillna(0) >= MIN_FLOAT]  # liquidity screen
    uni = uni.dropna(subset=["Sector"])                        # need a sector to be usable
    rets = rets[uni["Ticker"].tolist()]                        # keep matrix in sync

    with open(os.path.join(CACHE, "returns.pkl"), "wb") as f:
        pickle.dump(rets, f)
    uni.to_csv(os.path.join(CACHE, "stock_universe.csv"), index=False)

    print("\n" + "=" * 60)
    print(f"stock_universe: {len(uni)} qualified stocks "
          f"(from {before} priced, float>={MIN_FLOAT:.0%}, sector-mapped)")
    print(uni.sort_values("Avg_Turnover_LKR", ascending=False).head(10).to_string(index=False))
    print("\ncache/ written: stock_universe.csv, returns.pkl, sector_returns.csv")


if __name__ == "__main__":
    main()
