import os,requests
def format_signal(s):
    return (f"PAPER SIGNAL ONLY | {s['side']} {s['symbol']} (15m)\n"
      f"Entry {s['entry']:.4f} | Stop {s['stop']:.4f} | Target {s['target']:.4f}\n"
      f"Qty {s['qty']:.6f} | Notional {s['notional']:.2f} USDT\n"
      f"Risk budget {s['risk']:.2f} USDT | RSI {s['rsi']:.1f} | Volume {s['volume_ratio']:.2f}x\n"
      "No real order placed.")
def send(text):
    token=os.getenv("TELEGRAM_BOT_TOKEN");chat=os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:print("Telegram not configured; dry run:\n"+text);return False
    r=requests.post(f"https://api.telegram.org/bot{token}/sendMessage",json={"chat_id":chat,"text":text},timeout=20)
    r.raise_for_status();return True
