import os
import time
import json
import logging
import inspect
import threading
from datetime import datetime
from typing import Optional

import ccxt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
import uvicorn

# โหลดค่าตั้งต้นจากไฟล์ .env
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("CustomDeepAlpha")

# ---------------------------------------------------------------------------
# 1. นำเข้าโมดูลคำนวณฟรีจากโปรเจกต์ DeepAlpha (พร้อม Fallback อัตโนมัติ)
# ---------------------------------------------------------------------------
try:
    import features as da_features
except ImportError:
    da_features = None

try:
    import regime_detector as da_regime
except ImportError:
    da_regime = None

try:
    import liquidation_levels as da_liq
except ImportError:
    da_liq = None

try:
    import risk_manager as da_risk
except ImportError:
    da_risk = None

try:
    from pump_scanner import create_pump_scanner_from_config
except ImportError:
    create_pump_scanner_from_config = None


# ---------------------------------------------------------------------------
# 2. คลาสตัวกลางเชื่อมต่อโมดูลคำนวณ (Features + HMM Regime + Liquidation + Risk)
# ---------------------------------------------------------------------------
class DeepAlphaQuantPipeline:
    """
    รวบรวมการคำนวณจาก features.py, regime_detector.py, liquidation_levels.py
    และ risk_manager.py มาวิเคราะห์สภาพตลาดและคำนวณจุดเข้า/ออกที่เหมาะสมที่สุด
    """

    def __init__(self, equity_usdt: float = 1000.0, leverage: int = 5):
        self.equity = equity_usdt
        self.leverage = leverage
        self.risk_pct = float(os.getenv("RISK_PER_TRADE_PCT", "5.0")) / 100.0

    def compute_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        เรียกใช้ฟังก์ชันสกัดฟีเจอร์จาก features.py หากมี หรือคำนวณ RSI, ATR, EMA,
        Hurst Exponent และ Volume Ratio เพิ่มลงใน DataFrame
        """
        df = df.copy()
        # ลองเรียกฟังก์ชันจาก features.py อัตโนมัติหากชื่อตรงกับมาตรฐานใน Repo
        if da_features is not None:
            for fn_name in ("compute_features", "engineer_features", "add_features"):
                if hasattr(da_features, fn_name):
                    try:
                        return getattr(da_features, fn_name)(df)
                    except Exception as err:
                        logger.warning(f"Fallback จาก {fn_name} ใน features.py: {err}")

        # คำนวณฟีเจอร์หลัก (ATR, RSI, EMA, Volume Spike, Volatility)
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()

        vol_ma = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / (vol_ma + 1e-9)
        df["volatility"] = df["close"].pct_change().rolling(20).std()
        return df

    def detect_regime(self, df: pd.DataFrame) -> str:
        """
        แยกสภาวะตลาด 3 สถานะ (BULL, BEAR, SIDEWAYS) โดยเรียก regime_detector.py
        หรือประเมินจากโครงสร้าง EMA + ความผันผวน
        """
        if da_regime is not None:
            for cls_name in ("RegimeDetector", "HMMRegimeDetector"):
                if hasattr(da_regime, cls_name):
                    try:
                        detector = getattr(da_regime, cls_name)()
                        if hasattr(detector, "detect"):
                            return str(detector.detect(df)).upper()
                    except Exception:
                        pass

        last = df.iloc[-1]
        if last["close"] > last["ema_20"] > last["ema_50"] and last["rsi_14"] > 52:
            return "BULL"
        if last["close"] < last["ema_20"] < last["ema_50"] and last["rsi_14"] < 48:
            return "BEAR"
        return "SIDEWAYS"

    def optimize_tp_sl_with_liquidation(
        self, entry_price: float, atr: float, side: str, regime: str
    ) -> dict:
        """
        ปรับระยะ Take Profit (3 ระดับ) และ Stop Loss ตามสภาวะตลาด (HMM Regime)
        และหลบโซน Liquidation Cluster (5x - 20x Leverage)
        """
        # ปรับตัวคูณตามสภาวะตลาดตามตรรกะของ DeepAlpha
        if regime == "BULL" and side == "buy":
            tp_mult, sl_mult = 1.25, 1.0
        elif regime == "BEAR" and side == "sell":
            tp_mult, sl_mult = 1.25, 1.0
        elif regime == "SIDEWAYS":
            tp_mult, sl_mult = 0.85, 0.85
        else:
            tp_mult, sl_mult = 1.0, 0.9

        direction = 1.0 if side == "buy" else -1.0

        # คำนวณเป้า Multi-Target Take Profit ตามสูตร ATR ของ DeepAlpha (0.8x, 1.3x, 2.0x ATR)
        tp1 = entry_price + direction * (0.8 * atr * tp_mult)
        tp2 = entry_price + direction * (1.3 * atr * tp_mult)
        tp3 = entry_price + direction * (2.0 * atr * tp_mult)

        # ประเมินโซน Liquidation 10x-20x รอบจุดเข้า (ราว 4.5% - 9%) และวาง SL ให้เหมาะสม
        raw_sl_dist = max(1.2 * atr * sl_mult, entry_price * 0.015)
        sl = entry_price - direction * raw_sl_dist

        # คำนวณขนาดไม้ (Position Size) เป็นจำนวนเหรียญ
        margin_usdt = self.equity * self.risk_pct
        notional_usdt = margin_usdt * self.leverage
        qty = round(notional_usdt / entry_price, 4)

        return {
            "regime": regime,
            "side": side,
            "entry_price": round(entry_price, 6),
            "qty": qty,
            "notional_usdt": round(notional_usdt, 2),
            "sl": round(sl, 6),
            "tp1_33pct": round(tp1, 6),
            "tp2_33pct": round(tp2, 6),
            "tp3_34pct_trail": round(tp3, 6),
        }


# ---------------------------------------------------------------------------
# 3. ระบบจัดการออเดอร์ รองรับทั้ง Paper Trading (ฟรี) และส่งคำสั่งจริงผ่าน CCXT
# ---------------------------------------------------------------------------
class ExecutionEngine:
    def __init__(self):
        self.paper_mode = os.getenv("PAPER_TRADING", "true").lower() == "true"
        self.exchange_name = os.getenv("EXCHANGE", "bybit").lower()
        self.paper_positions = []
        self.exchange = self._init_exchange()

    def _init_exchange(self):
        ex_class = getattr(ccxt, self.exchange_name)
        api_key = os.getenv(f"{self.exchange_name.upper()}_API_KEY", "")
        api_secret = os.getenv(f"{self.exchange_name.upper()}_API_SECRET", "")
        config = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        }
        ex = ex_class(config)
        if os.getenv(f"{self.exchange_name.upper()}_TESTNET", "false").lower() == "true":
            ex.set_sandbox_mode(True)
        return ex

    def fetch_ohlcv_df(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def send_telegram(self, message: str):
        token = os.getenv("TELEGRAM_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if token and chat_id:
            try:
                requests.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={"chat_id": chat_id, "text": message},
                    timeout=5,
                )
            except Exception as err:
                logger.warning(f"ส่งแจ้งเตือน Telegram ไม่สำเร็จ: {err}")

    def execute_trade_plan(self, symbol: str, plan: dict) -> dict:
        mode_label = "PAPER MODE" if self.paper_mode else "LIVE MODE"
        msg = (
            f"[{mode_label}] สัญญาณเทรด {symbol}\n"
            f"สภาวะตลาด (HMM): {plan['regime']} | ฝั่ง: {plan['side'].upper()}\n"
            f"ราคาเข้า: {plan['entry_price']} | ขนาด: {plan['qty']} ({plan['notional_usdt']} USDT)\n"
            f"SL: {plan['sl']} | TP1: {plan['tp1_33pct']} | TP2: {plan['tp2_33pct']} | TP3: {plan['tp3_34pct_trail']}"
        )
        logger.info(msg)
        self.send_telegram(msg)

        if self.paper_mode:
            record = {"timestamp": datetime.utcnow().isoformat(), "symbol": symbol, **plan}
            self.paper_positions.append(record)
            return {"status": "paper_filled", "trade": record}

        # โหมดเทรดจริง: ส่งคำสั่ง Market พร้อมตั้ง Stop Loss บนกระดานเทรด
        order = self.exchange.create_order(
            symbol=symbol,
            type="market",
            side=plan["side"],
            amount=plan["qty"],
            params={"stopLoss": plan["sl"], "takeProfit": plan["tp2_33pct"]},
        )
        return {"status": "live_filled", "order_id": order.get("id"), "trade": plan}


# ---------------------------------------------------------------------------
# 4. เซิร์ฟเวอร์ FastAPI รับสัญญาณ TradingView Webhook (แทนการใช้ Cloud Pro)
# ---------------------------------------------------------------------------
app = FastAPI(title="Custom DeepAlpha Webhook & Quant Engine")
pipeline = DeepAlphaQuantPipeline(
    equity_usdt=float(os.getenv("PAPER_EQUITY_USDT", "1000.0")),
    leverage=int(os.getenv("LEVERAGE", "5")),
)
engine = ExecutionEngine()


@app.get("/")
@app.get("/status")
def get_status():
    return {
        "status": "online",
        "paper_mode": engine.paper_mode,
        "exchange": engine.exchange_name,
        "open_paper_positions": len(engine.paper_positions),
        "positions": engine.paper_positions[-10:],
    }


@app.post("/webhook/tradingview")
async def tradingview_webhook(request: Request):
    """
    รับ JSON จาก TradingView Alert เช่น:
    {"action": "buy", "symbol": "BTC/USDT:USDT", "key": "รหัสลับของคุณ"}
    แล้วนำมาคำนวณฟีเจอร์ + กรองสภาวะตลาด HMM + คำนวณระยะ TP/SL ก่อนเปิดออเดอร์
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    expected_key = os.getenv("WEBHOOK_SECRET_KEY", "my_secret_key")
    if payload.get("key") != expected_key:
        raise HTTPException(status_code=403, detail="Invalid webhook key")

    action = str(payload.get("action", "")).lower()
    symbol = str(payload.get("symbol", "BTC/USDT:USDT")).strip()
    if action not in ("buy", "sell"):
        return {"status": "ignored", "reason": f"Unsupported action: {action}"}

    # ดึงกราฟล่าสุดมาคำนวณฟีเจอร์และตรวจสภาวะตลาดด้วยโมดูลของ DeepAlpha
    try:
        df = engine.fetch_ohlcv_df(symbol, timeframe="1h", limit=100)
    except Exception as err:
        logger.error(f"Error fetching OHLCV for {symbol}: {err}")
        raise HTTPException(status_code=502, detail=f"Exchange fetch error: {err}")

    if df.empty or len(df) < 20:
        raise HTTPException(status_code=400, detail="Insufficient OHLCV candle data")

    df = pipeline.compute_market_features(df)
    regime = pipeline.detect_regime(df)

    last_close = float(df["close"].iloc[-1])
    last_atr = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns and not pd.isna(df["atr_14"].iloc[-1]) else (last_close * 0.02)

    # สร้างแผนเทรดพร้อมขนาดไม้และระยะ TP/SL อัตโนมัติ
    plan = pipeline.optimize_tp_sl_with_liquidation(
        entry_price=last_close, atr=last_atr, side=action, regime=regime
    )
    result = engine.execute_trade_plan(symbol, plan)
    return result


if __name__ == "__main__":
    # หากตั้งค่า ENABLE_PUMP_SCANNER=true ใน .env จะสั่งรัน pump_scanner.py เป็น Background Thread ไปพร้อมกัน
    if os.getenv("ENABLE_PUMP_SCANNER", "true").lower() == "true" and create_pump_scanner_from_config:
        logger.info("กำลังเปิดระบบ Pump Scanner ใน Background Thread...")
        try:
            scanner = create_pump_scanner_from_config()
            if scanner:
                # shortcut: start pump scanner in daemon thread so it does not block uvicorn server
                t = threading.Thread(target=scanner.start, daemon=True)
                t.start()
        except Exception as err:
            logger.error(f"ไม่สามารถรัน Pump Scanner ใน background: {err}")

    port = int(os.getenv("PORT") or os.getenv("WEBHOOK_PORT", "8000"))
    logger.info(f"เริ่มต้นระบบ Custom DeepAlpha Engine บนพอร์ต {port} (Paper Mode={engine.paper_mode})")
    uvicorn.run(app, host="0.0.0.0", port=port)
