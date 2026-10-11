import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime

st.set_page_config(page_title="Midcap Momentum Breakout", page_icon="📈", layout="wide")

MIDCAP_URLS = {
    "Nifty Midcap 100": "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
    "Nifty Midcap 150": "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
}

st.title("📈 Midcap Momentum Breakout Dashboard")
st.caption("Daily trend + Keltner breakout + CMO + volume | Research and paper-trading tool")

with st.sidebar:
    st.header("Scanner settings")
    universe_name = st.selectbox("Universe", list(MIDCAP_URLS.keys()), index=1)
    period = st.selectbox("Historical data", ["2y", "3y", "5y"], index=2)
    roe_min = st.number_input("Minimum ROE (%)", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
    debt_max = st.number_input("Maximum Debt/Equity", min_value=0.0, max_value=10.0, value=0.8, step=0.1)
    use_fundamentals = st.checkbox("Apply fundamental filters (when Yahoo data available)", value=True)
    atr_mult = st.number_input("ATR stop multiplier", min_value=0.5, max_value=5.0, value=1.5, step=0.25)
    target_r = st.selectbox("Target multiple", [1.5, 2.0, 2.5, 3.0], index=1)
    risk_pct = st.number_input("Risk budget per trade (%)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
    capital = st.number_input("Capital for sizing (₹)", min_value=10000, value=200000, step=10000)
    run_scan = st.button("Run scan & backtest", type="primary", use_container_width=True)

st.info("Important: this version uses daily candles and confirms signals from completed daily bars. A 3:15 PM scan is not equivalent to a confirmed daily close. Treat results as research, not financial advice.")

def get_universe(label):
    url = MIDCAP_URLS[label]
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    col = next((c for c in ["Symbol", "symbol", "SYMBOL"] if c in df.columns), None)
    if not col:
        raise ValueError("Could not identify Symbol column in index constituent file.")
    syms = df[col].astype(str).str.strip().str.upper()
    return sorted(set(s + ".NS" for s in syms if s and s != "NAN"))

@st.cache_data(ttl=3600, show_spinner=False)
def load_history(ticker, period):
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()
    df = df[needed].dropna().copy()
    df.index = pd.to_datetime(df.index)
    return df

def indicators(df):
    d = df.copy()
    d["EMA20"] = d["Close"].ewm(span=20, adjust=False).mean()
    d["EMA150"] = d["Close"].ewm(span=150, adjust=False).mean()
    prev_close = d["Close"].shift(1)
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - prev_close).abs(),
        (d["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    d["ATR20"] = tr.rolling(20).mean()
    d["ATR14"] = tr.rolling(14).mean()
    d["KeltnerUpper"] = d["EMA20"] + 1.5 * d["ATR20"]
    # Chande Momentum Oscillator (9): 100*(sum(up)-sum(down))/(sum(up)+sum(down))
    delta = d["Close"].diff()
    up = delta.clip(lower=0).rolling(9).sum()
    down = (-delta.clip(upper=0)).rolling(9).sum()
    denom = up + down
    d["CMO9"] = np.where(denom == 0, 0, 100 * (up - down) / denom)
    d["VolAvg20"] = d["Volume"].rolling(20).mean()
    d["Signal"] = (
        (d["Close"] > d["EMA150"]) &
        (d["Close"] > d["KeltnerUpper"]) &
        (d["CMO9"] > 0) & (d["CMO9"].shift(1) <= 0) &
        (d["Volume"] > 2 * d["VolAvg20"])
    )
    return d

def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        roe = info.get("returnOnEquity")
        debt = info.get("debtToEquity")
        roe_pct = float(roe) * 100 if roe is not None else np.nan
        debt_ratio = float(debt) / 100 if debt is not None else np.nan
        return roe_pct, debt_ratio
    except Exception:
        return np.nan, np.nan

def backtest(d, atr_mult, target_r, max_hold=20):
    # Signals are generated on the completed signal-day close. Entry uses next day's open.
    # If stop and target occur in the same daily bar, assume stop is hit first (conservative).
    rows = []
    sig_positions = np.flatnonzero(d["Signal"].fillna(False).values)
    for i in sig_positions:
        if i + 1 >= len(d):
            continue
        entry_day = i + 1
        entry = float(d["Open"].iloc[entry_day])
        atr = float(d["ATR14"].iloc[i])
        if not np.isfinite(entry) or not np.isfinite(atr) or atr <= 0:
            continue
        stop = entry - atr_mult * atr
        if stop <= 0:
            continue
        risk = entry - stop
        target = entry + target_r * risk
        exit_price, exit_date, reason = None, None, "Time exit"
        end = min(len(d) - 1, entry_day + max_hold - 1)
        for j in range(entry_day, end + 1):
            low, high = float(d["Low"].iloc[j]), float(d["High"].iloc[j])
            if low <= stop:
                exit_price, exit_date, reason = stop, d.index[j], "Stop-loss"
                break
            if high >= target:
                exit_price, exit_date, reason = target, d.index[j], "Target"
                break
        if exit_price is None:
            exit_price, exit_date = float(d["Close"].iloc[end]), d.index[end]
        pnl_pct = (exit_price / entry - 1) * 100
        r_multiple = (exit_price - entry) / risk
        rows.append({
            "Signal Date": d.index[i].date().isoformat(),
            "Entry Date": d.index[entry_day].date().isoformat(),
            "Entry": entry, "Stop": stop, "Target": target,
            "Exit Date": pd.Timestamp(exit_date).date().isoformat(),
            "Exit": exit_price, "Exit Reason": reason,
            "Return %": pnl_pct, "R Multiple": r_multiple
        })
    return pd.DataFrame(rows)

def scan_ticker(ticker, period, use_fundamentals, roe_min, debt_max, atr_mult, target_r):
    hist = load_history(ticker, period)
    if hist.empty or len(hist) < 180:
        return None, pd.DataFrame(), "Insufficient price history"
    d = indicators(hist)
    last = d.iloc[-1]
    if not np.isfinite(last["ATR14"]) or not np.isfinite(last["KeltnerUpper"]):
        return None, pd.DataFrame(), "Indicators not ready"
    roe, debt = (np.nan, np.nan)
    if use_fundamentals:
        roe, debt = get_fundamentals(ticker)
        # Unknown fundamentals are flagged rather than silently passing.
        fundamental_ok = pd.notna(roe) and pd.notna(debt) and roe > roe_min and debt < debt_max
    else:
        fundamental_ok = True
    technical_count = sum([
        bool(last["Close"] > last["EMA150"]),
        bool(last["Close"] > last["KeltnerUpper"]),
        bool(last["CMO9"] > 0 and d["CMO9"].iloc[-2] <= 0),
        bool(last["Volume"] > 2 * last["VolAvg20"])
    ])
    entry = float(last["Close"])
    stop = entry - atr_mult * float(last["ATR14"])
    target = entry + target_r * (entry - stop)
    return {
        "Symbol": ticker.replace(".NS", ""),
        "Date": d.index[-1].date().isoformat(),
        "Close (₹)": entry,
        "EMA150 (₹)": float(last["EMA150"]),
        "Keltner Upper (₹)": float(last["KeltnerUpper"]),
        "CMO(9)": float(last["CMO9"]),
        "Volume": int(last["Volume"]),
        "20D Avg Volume": float(last["VolAvg20"]),
        "Technical Checks": f"{technical_count}/4",
        "ROE (%)": roe,
        "Debt/Equity": debt,
        "Fundamentals": "Pass" if fundamental_ok else ("Unavailable" if pd.isna(roe) or pd.isna(debt) else "Fail"),
        "Signal": "BUY SETUP" if bool(last["Signal"]) and fundamental_ok else ("TECHNICAL ONLY" if bool(last["Signal"]) else "No signal"),
        "Indicative Stop (₹)": stop,
        f"Target ({target_r}R) (₹)": target,
        "ATR14 (₹)": float(last["ATR14"])
    }, backtest(d, atr_mult, target_r), None

if not run_scan:
    st.subheader("What the dashboard will show")
    c1, c2, c3 = st.columns(3)
    c1.metric("Technical rules", "4")
    c2.metric("Risk model", f"{atr_mult} × ATR(14)")
    c3.metric("Target", f"{target_r}R")
    st.markdown("""
    **Scanner columns**
    - Stock, close, EMA150, Keltner upper band, CMO(9), volume vs. 20-day average
    - Technical checks, ROE, debt/equity, signal status, ATR stop and target

    **Validation metrics**
    - Win rate, profit factor, average R/trade, total return, max drawdown
    - Trade-by-trade entry/exit log, with conservative stop-first treatment when stop and target are both touched in one daily candle
    """)
    st.stop()

try:
    with st.spinner("Loading index constituents..."):
        tickers = get_universe(universe_name)
except Exception as e:
    st.error(f"Could not load the official index constituent list: {e}")
    st.markdown("Check your internet connection or download the constituent CSV from the Nifty Indices website and retry later.")
    st.stop()

st.write(f"Universe loaded: **{len(tickers)} symbols**. Downloading daily history and screening may take several minutes.")
scan_rows, all_trades, errors = [], [], []
progress = st.progress(0)
status = st.empty()
for n, ticker in enumerate(tickers):
    status.text(f"Processing {ticker} ({n+1}/{len(tickers)})")
    try:
        row, trades, err = scan_ticker(ticker, period, use_fundamentals, roe_min, debt_max, atr_mult, target_r)
        if row:
            scan_rows.append(row)
            if not trades.empty:
                trades.insert(0, "Symbol", ticker.replace(".NS", ""))
                all_trades.append(trades)
        elif err:
            errors.append((ticker, err))
    except Exception as e:
        errors.append((ticker, str(e)[:120]))
    progress.progress((n + 1) / len(tickers))
status.empty()
progress.empty()

if not scan_rows:
    st.warning("No usable stock data was returned. Check Yahoo Finance availability and try again.")
    st.stop()

scan = pd.DataFrame(scan_rows)
scan = scan.sort_values(["Signal", "Technical Checks"], ascending=[True, False])
signals = scan[scan["Signal"].isin(["BUY SETUP", "TECHNICAL ONLY"])].copy()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Stocks screened", len(scan))
m2.metric("Buy setups", int((scan["Signal"] == "BUY SETUP").sum()))
m3.metric("Technical-only signals", int((scan["Signal"] == "TECHNICAL ONLY").sum()))
m4.metric("Data issues", len(errors))
st.subheader("Scanner results")
filter_choice = st.selectbox("Show", ["All stocks", "Signals only", "Technical checks 4/4"])
view = scan.copy()
if filter_choice == "Signals only":
    view = view[view["Signal"].isin(["BUY SETUP", "TECHNICAL ONLY"])]
elif filter_choice == "Technical checks 4/4":
    view = view[view["Technical Checks"] == "4/4"]
st.dataframe(view, use_container_width=True, hide_index=True)
st.download_button("Download scanner CSV", scan.to_csv(index=False).encode("utf-8"), "midcap_scan_results.csv", "text/csv")

trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
st.subheader("Historical validation")
if trades.empty:
    st.warning("No completed historical trades were generated from available signals. This is not evidence of profitability.")
else:
    win_rate = (trades["R Multiple"] > 0).mean() * 100
    gross_profit = trades.loc[trades["Return %"] > 0, "Return %"].sum()
    gross_loss = abs(trades.loc[trades["Return %"] < 0, "Return %"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.nan
    expectancy = trades["R Multiple"].mean()
    # Portfolio-level equity curve is not simulated; drawdown below is per-trade equal-weight sequence, sorted by exit date.
    ordered = trades.sort_values("Exit Date").copy()
    ordered["Equity"] = (1 + ordered["Return %"] / 100).cumprod()
    peak = ordered["Equity"].cummax()
    max_dd = ((ordered["Equity"] / peak) - 1).min() * 100
    a, b, c, dcol, e = st.columns(5)
    a.metric("Closed trades", len(trades))
    b.metric("Win rate", f"{win_rate:.1f}%")
    c.metric("Profit factor", f"{pf:.2f}" if np.isfinite(pf) else "N/A")
    dcol.metric("Expectancy", f"{expectancy:.2f}R")
    e.metric("Trade-sequence drawdown", f"{max_dd:.1f}%")
    st.caption("Initial, overlapping trade-by-trade test; equal-weight compounding is illustrative and does not model portfolio concurrency, slippage, brokerage, taxes, corporate actions, or capital constraints. Do not interpret it as a validated live strategy.")
    st.dataframe(trades.sort_values("Entry Date", ascending=False), use_container_width=True, hide_index=True)
    st.download_button("Download trade log CSV", trades.to_csv(index=False).encode("utf-8"), "midcap_backtest_trades.csv", "text/csv")

if errors:
    with st.expander(f"Data issues ({len(errors)})"):
        st.dataframe(pd.DataFrame(errors, columns=["Symbol", "Issue"]), use_container_width=True, hide_index=True)

st.caption(f"Last run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} local machine time. Data and fundamentals depend on third-party availability and may be delayed or incomplete.")
