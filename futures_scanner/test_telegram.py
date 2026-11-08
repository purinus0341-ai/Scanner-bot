from .notify import send
if __name__ == "__main__":
    if not send("Scanner-bot Telegram connection test OK. No trade was placed."):
        raise SystemExit("Telegram credentials not configured")
    print("Telegram test message sent")
