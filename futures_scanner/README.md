# Crypto Futures Scanner — read-only
Isolated from existing custom_bot_engine.py. No order placement, API trading keys, or withdrawals.
## Setup
Configure repository Actions secrets TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID. Enable GitHub Actions, review workflow permissions (contents write needed for dedup state), and merge only after review. Scheduled workflows run from the default branch, so the schedule does not run on this feature branch.
## Commands
pip install -r futures_scanner/requirements.txt
python -m pytest futures_scanner/tests -q
python -m futures_scanner.main
python -m futures_scanner.backtest --symbol BTCUSDT --bars 2000
## Limits
50 USDT account, theoretical risk 0.5 USDT per signal, BTCUSDT and ETHUSDT 15m entries / 1h trend. No live execution. No guaranteed profitability. Backtest is simplified: does not model funding, partial fills, or exchange minimum orders; a signal is not an instruction to trade. Scheduled jobs can be delayed. Public Binance Futures endpoints may be blocked in some jurisdictions. Keep secrets out of code.

## Market data provider (November 2026)
Binance Futures `fapi.binance.com` returned HTTP 451 on GitHub-hosted runner. The scanner now reads **Bybit public USDT perpetual** candles from `/v5/market/kline` (15-minute and 60-minute). This is a **different exchange** with different prices, volume and fees; historical results and alerts are not directly comparable to Binance. It does not trade on either exchange. If Bybit also blocks access, the workflow fails clearly rather than bypassing restrictions or inventing signals. Verify local eligibility and exchange terms. GitHub-hosted network access has not yet been live-tested after this change.


## Hyperliquid perpetual market data
Binance Futures returned HTTP 451 and Bybit returned HTTP 403 on GitHub-hosted runners. The read-only data source is now Hyperliquid public perpetual candles via POST /info, type candleSnapshot, for BTC and ETH. **Hyperliquid perpetuals are USDC-margined; BTCUSDT and ETHUSDT are retained as internal scanner aliases only, not exchange contract identifiers.** Do not treat these signals as Binance/Bybit prices or as executable orders. Risk and notional amounts are illustrative USD-equivalent figures; they are not actual USDT wallet balances. The historical endpoint has a rolling 5,000-candle window. This change has not yet been verified by a live GitHub Actions run; if the endpoint is blocked, it fails without fabricating signals. Review venue eligibility and terms before use.
