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