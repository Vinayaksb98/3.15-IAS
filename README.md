# Midcap Momentum Breakout Dashboard

A Streamlit research dashboard for the proposed Nifty Midcap 100/150 daily breakout rules.

## Rules implemented

- Close > EMA(150)
- Close > upper Keltner Channel: EMA(20) + 1.5 × ATR(20)
- CMO(9) crosses above zero
- Volume > 2 × 20-day average volume
- Optional fundamental gate: ROE > 15% and Debt/Equity < 0.8
- Initial stop: entry − 1.5 × ATR(14)
- Target: configurable R multiple (default 2R)

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Start the dashboard:

   `streamlit run app.py`

## Deploy on Streamlit Community Cloud

1. Upload these files to a GitHub repository.
2. Choose `app.py` as the main file.
3. Streamlit Cloud installs packages from `requirements.txt`.

## Important limitations

- This is a starter implementation, not a certified or independently validated trading system.
- The index constituent CSV and Yahoo Finance data require internet access and may change or be unavailable.
- Yahoo Finance fundamental fields may be missing. If the fundamental gate is enabled, missing ROE/debt data is flagged as unavailable and does not pass the gate.
- Daily data confirms the signal only after a daily candle completes. A scan at 3:15 PM IST cannot confirm the final daily close or final daily volume.
- Historical entries use the next available daily open after a signal. If stop and target are both touched in one candle, the backtest assumes the stop was hit first.
- The simple trade log can contain overlapping trades and does not model portfolio-level capital allocation, partial fills, brokerage, taxes, slippage, dividends, splits, or gaps beyond the assumed stop price. The displayed trade-sequence drawdown is illustrative, not a portfolio backtest.
- No claim of profitability is made. Validate out-of-sample and paper-trade before using real money.

## Suggested next improvements

- Add intraday volume-at-time normalization for a true 3:15 PM scan.
- Add point-in-time historical index constituents to avoid survivorship bias.
- Add portfolio-level trade concurrency, costs, slippage, and risk-budget simulation.
- Compare rule variants on a separate out-of-sample period.
