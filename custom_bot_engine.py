import os
import time
import json
import asyncio
import logging
from datetime import datetime
from typing import Optional

import ccxt
import ccxt.async_support as ccxt_async
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


# ---------------------------------------------------------------------------
# 2. ระบบจัดเก็บข้อมูลถาวรฟรี 100% (Telegram Cloud Backup + Supabase Free Tier)
# ---------------------------------------------------------------------------
class FreePersistentStorage:
    def __init__(self):
        self.tg_token = os.getenv("TELEGRAM_TOKEN", "")
        self.tg_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.supa_url = os.getenv("SUPABASE_URL", "").rstrip("/")
        self.supa_key = os.getenv("SUPABASE_KEY", "")
        self.supa_enabled = bool(self.supa_url and self.supa_key)
        self.headers = {
            "apikey": self.supa_key,
            "Authorization": f"Bearer {self.supa_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }

    def save_state(self, positions: list, realized_pnl: float, history: list):
        if self.supa_enabled:
            try:
                payload = {
                    "id": "main_portfolio",
                    "updated_at": datetime.utcnow().isoformat(),
                    "realized_pnl": realized_pnl,
                    "open_positions": positions,
                    "closed_history": history[-30:],
                }
                requests.post(
                    f"{self.supa_url}/rest/v1/bot_state?on_conflict=id",
                    headers={**self.headers, "Prefer": "resolution=merge-duplicates"},
                    json=payload,
                    timeout=5,
                )
            except Exception as err:
                logger.debug(f"Supabase save warning: {err}")

        if self.tg_token and self.tg_chat_id:
            try:
                state_json = json.dumps({
                    "pnl": round(realized_pnl, 2),
                    "open": positions,
                    "history": history[-10:],
                })
                text = f"📦 [BOT_STATE_BACKUP]\n{state_json}"
                chat_res = requests.get(
                    f"https://api.telegram.org/bot{self.tg_token}/getChat",
                    params={"chat_id": self.tg_chat_id},
                    timeout=5,
                ).json()
                pinned = chat_res.get("result", {}).get("pinned_message", {})
                pinned_id = pinned.get("message_id")

                if pinned_id and "[BOT_STATE_BACKUP]" in pinned.get("text", ""):
                    requests.post(
                        f"https://api.telegram.org/bot{self.tg_token}/editMessageText",
                        json={
                            "chat_id": self.tg_chat_id,
                            "message_id": pinned_id,
                            "text": text,
                        },
                        timeout=5,
                    )
                else:
                    send_res = requests.post(
                        f"https://api.telegram.org/bot{self.tg_token}/sendMessage",
                        json={"chat_id": self.tg_chat_id, "text": text, "disable_notification": True},
                        timeout=5,
                    ).json()
                    msg_id = send_res.get("result", {}).get("message_id")
                    if msg_id:
                        requests.post(
                            f"https://api.telegram.org/bot{self.tg_token}/pinChatMessage",
                            json={"chat_id": self.tg_chat_id, "message_id": msg_id, "disable_notification": True},
                            timeout=5,
                        )
            except Exception as err:
                logger.debug(f"Telegram state backup warning: {err}")

    def load_state(self) -> tuple[list, float, list]:
        if self.supa_enabled:
            try:
                res = requests.get(
                    f"{self.supa_url}/rest/v1/bot_state?id=eq.main_portfolio",
                    headers=self.headers,
                    timeout=5,
                )
                if res.status_code == 200 and res.json():
                    data = res.json()[0]
                    return (
                        data.get("open_positions", []),
                        float(data.get("realized_pnl", 0.0)),
                        data.get("closed_history", []),
                    )
            except Exception as err:
                logger.debug(f"Supabase load warning: {err}")

        if self.tg_token and self.tg_chat_id:
            try:
                chat_res = requests.get(
                    f"https://api.telegram.org/bot{self.tg_token}/getChat",
                    params={"chat_id": self.tg_chat_id},
                    timeout=5,
                ).json()
                pinned_text = chat_res.get("result", {}).get("pinned_message", {}).get("text", "")
                if "[BOT_STATE_BACKUP]" in pinned_text:
                    raw_json = pinned_text.split("[BOT_STATE_BACKUP]\n", 1)[1]
                    data = json.loads(raw_json)
                    return (
                        data.get("open", []),
                        float(data.get("pnl", 0.0)),
                        data.get("history", []),
                    )
            except Exception as err:
                logger.debug(f"Telegram state restore warning: {err}")

        return [], 0.0, []


# ---------------------------------------------------------------------------
# 3. คลาสคำนวณเชิงปริมาณ (Features + HMM Regime + Liquidation + Alpha Filter)
# ---------------------------------------------------------------------------
class DeepAlphaQuantPipeline:
    def __init__(self, equity_usdt: float = 1000.0, leverage: int = 5):
        self.equity = equity_usdt
        self.leverage = leverage
        self.risk_pct = float(os.getenv("RISK_PER_TRADE_PCT", "5.0")) / 100.0

    def compute_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if da_features is not None:
            for fn_name in ("compute_features", "engineer_features", "add_features"):
                if hasattr(da_features, fn_name):
                    try:
                        return getattr(da_features, fn_name)(df)
                    except Exception:
                        pass

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
        self, entry_price: float, atr: float, side: str, regime: str, current_equity: float
    ) -> dict:
        if regime == "BULL" and side == "buy":
            tp_mult, sl_mult = 1.25, 1.0
        elif regime == "BEAR" and side == "sell":
            tp_mult, sl_mult = 1.25, 1.0
        elif regime == "SIDEWAYS":
            tp_mult, sl_mult = 0.85, 0.85
        else:
            tp_mult, sl_mult = 1.0, 0.9

        direction = 1.0 if side == "buy" else -1.0

        tp1 = entry_price + direction * (0.8 * atr * tp_mult)
        tp2 = entry_price + direction * (1.3 * atr * tp_mult)
        tp3 = entry_price + direction * (2.0 * atr * tp_mult)

        raw_sl_dist = max(1.2 * atr * sl_mult, entry_price * 0.015)
        sl = entry_price - direction * raw_sl_dist

        margin_usdt = max(current_equity, 100.0) * self.risk_pct
        notional_usdt = margin_usdt * self.leverage
        qty = round(notional_usdt / entry_price, 4)

        return {
            "regime": regime,
            "side": side,
            "entry_price": round(entry_price, 6),
            "qty": qty,
            "remaining_qty": qty,
            "notional_usdt": round(notional_usdt, 2),
            "sl": round(sl, 6),
            "initial_sl": round(sl, 6),
            "tp1_33pct": round(tp1, 6),
            "tp2_33pct": round(tp2, 6),
            "tp3_34pct_trail": round(tp3, 6),
            "tp1_hit": False,
            "tp2_hit": False,
            "realized_pnl": 0.0,
        }


# ---------------------------------------------------------------------------
# 4. ระบบจัดการออเดอร์และตรวจจับการชน TP/SL (Paper Exit Engine)
# ---------------------------------------------------------------------------
class ExecutionEngine:
    def __init__(self, storage: FreePersistentStorage):
        self.paper_mode = os.getenv("PAPER_TRADING", "true").lower() == "true"
        self.exchange_name = os.getenv("EXCHANGE", "gate").lower()
        self.initial_equity = float(os.getenv("PAPER_EQUITY_USDT", "1000.0"))
        self.storage = storage
        self.paper_positions, self.realized_total_pnl, self.closed_history = self.storage.load_state()
        self.exchange = self._init_sync_exchange()

    def _init_sync_exchange(self):
        ex_class = getattr(ccxt, self.exchange_name)
        config = {
            "apiKey": os.getenv(f"{self.exchange_name.upper()}_API_KEY", ""),
            "secret": os.getenv(f"{self.exchange_name.upper()}_API_SECRET", ""),
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        }
        return ex_class(config)

    @property
    def current_equity(self) -> float:
        return round(self.initial_equity + self.realized_total_pnl, 2)

    def send_telegram(self, message: str, reply_markup: Optional[dict] = None):
        token = os.getenv("TELEGRAM_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if token and chat_id:
            try:
                payload = {"chat_id": chat_id, "text": message}
                if reply_markup:
                    payload["reply_markup"] = reply_markup
                requests.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json=payload,
                    timeout=5,
                )
            except Exception as err:
                logger.warning(f"ส่งแจ้งเตือน Telegram ไม่สำเร็จ: {err}")

    def execute_trade_plan(self, symbol: str, plan: dict) -> dict:
        mode_label = "PAPER MODE" if self.paper_mode else "LIVE MODE"
        msg = (
            f"🚀 [{mode_label}] เปิดสัญญาณเทรด {symbol}\n"
            f"สภาวะตลาด (HMM): {plan['regime']} | ฝั่ง: {plan['side'].upper()}\n"
            f"ราคาเข้า: {plan['entry_price']} | ขนาด: {plan['qty']} ({plan['notional_usdt']} USDT)\n"
            f"SL: {plan['sl']} | TP1: {plan['tp1_33pct']} | TP2: {plan['tp2_33pct']} | TP3: {plan['tp3_34pct_trail']}"
        )
        logger.info(msg)
        self.send_telegram(msg)

        if self.paper_mode:
            record = {
                "id": f"{symbol.replace('/', '')}_{int(time.time())}",
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                **plan,
            }
            self.paper_positions.append(record)
            self.storage.save_state(self.paper_positions, self.realized_total_pnl, self.closed_history)
            return {"status": "paper_filled", "trade": record}

        order = self.exchange.create_order(
            symbol=symbol,
            type="market",
            side=plan["side"],
            amount=plan["qty"],
            params={"stopLoss": plan["sl"], "takeProfit": plan["tp2_33pct"]},
        )
        return {"status": "live_filled", "order_id": order.get("id"), "trade": plan}

    async def manual_close_position(self, target_symbol: str, exchange_async) -> str:
        """ฟังก์ชันปิดไม้ด้วยมือผ่านคำสั่ง Telegram"""
        target_upper = target_symbol.upper().strip()
        matched_pos = None
        for pos in self.paper_positions:
            sym = pos["symbol"].upper()
            if sym == target_upper or sym.startswith(target_upper + "/"):
                matched_pos = pos
                break

        if not matched_pos:
            return f"⚠️ ไม่พบไม้ที่เปิดอยู่สำหรับเหรียญ `{target_symbol}`"

        symbol = matched_pos["symbol"]
        try:
            ticker = await exchange_async.fetch_ticker(symbol)
            current_price = float(ticker["last"])
        except Exception:
            current_price = float(matched_pos.get("last_price") or matched_pos["entry_price"])

        entry = float(matched_pos["entry_price"])
        rem_qty = float(matched_pos.get("remaining_qty", matched_pos["qty"]))
        realized = float(matched_pos.get("realized_pnl", 0.0))
        is_buy = (matched_pos["side"] == "buy")

        pnl = (current_price - entry) * rem_qty if is_buy else (entry - current_price) * rem_qty
        realized += pnl
        self.realized_total_pnl += pnl

        matched_pos["remaining_qty"] = 0.0
        matched_pos["realized_pnl"] = round(realized, 2)
        matched_pos["closed_at"] = datetime.utcnow().isoformat()
        matched_pos["exit_reason"] = f"สั่งปิดด้วยมือผ่าน Telegram ที่ {current_price}"
        matched_pos["unrealized_pnl"] = 0.0

        self.paper_positions.remove(matched_pos)
        self.closed_history.append(matched_pos)
        self.storage.save_state(self.paper_positions, self.realized_total_pnl, self.closed_history)

        return (
            f"✅ ปิดไม้ {symbol} สำเร็จที่ราคา {current_price}\n"
            f"PnL สุทธิไม้นี้: {realized:+.2f} USDT\n"
            f"💰 พอร์ตจำลองปัจจุบัน: {self.current_equity:.2f} USDT"
        )

    async def close_all_positions(self, exchange_async) -> str:
        """ปิดทุกไม้ในพอร์ตทันทีผ่านคำสั่ง Telegram"""
        if not self.paper_positions:
            return "ℹ️ ปัจจุบันไม่มีไม้ที่เปิดค้างอยู่ในพอร์ตครับ"

        symbols = [pos["symbol"] for pos in list(self.paper_positions)]
        reports = []
        for sym in symbols:
            res = await self.manual_close_position(sym, exchange_async)
            reports.append(res)
        return "🚨 สรุปการปิดทุกไม้ทันที:\n\n" + "\n\n".join(reports)

    async def check_paper_exits(self, exchange_async):
        if not self.paper_positions:
            return

        symbols = list({pos["symbol"] for pos in self.paper_positions})
        try:
            tickers = await exchange_async.fetch_tickers(symbols)
        except Exception as err:
            logger.warning(f"ดึงราคาเช็ค TP/SL ไม่สำเร็จ: {err}")
            return

        state_changed = False
        remaining_positions = []

        for pos in self.paper_positions:
            symbol = pos["symbol"]
            if symbol not in tickers:
                remaining_positions.append(pos)
                continue

            current_price = float(tickers[symbol]["last"])
            pos["last_price"] = current_price
            side = pos["side"]
            entry = float(pos["entry_price"])
            sl = float(pos["sl"])
            tp1 = float(pos["tp1_33pct"])
            tp2 = float(pos["tp2_33pct"])
            tp3 = float(pos["tp3_34pct_trail"])

            total_qty = float(pos["qty"])
            rem_qty = float(pos.get("remaining_qty", total_qty))
            realized = float(pos.get("realized_pnl", 0.0))
            is_buy = (side == "buy")

            unrealized = (current_price - entry) * rem_qty if is_buy else (entry - current_price) * rem_qty
            pos["unrealized_pnl"] = round(unrealized, 2)

            closed_all = False
            exit_reason = ""

            if (is_buy and current_price <= sl) or (not is_buy and current_price >= sl):
                pnl = (current_price - entry) * rem_qty if is_buy else (entry - current_price) * rem_qty
                realized += pnl
                self.realized_total_pnl += pnl
                rem_qty = 0.0
                closed_all = True
                exit_reason = f"ชน SL ที่ {current_price}"

            elif (is_buy and current_price >= tp3) or (not is_buy and current_price <= tp3):
                pnl = (current_price - entry) * rem_qty if is_buy else (entry - current_price) * rem_qty
                realized += pnl
                self.realized_total_pnl += pnl
                rem_qty = 0.0
                closed_all = True
                exit_reason = f"ชนเป้าสูงสุด TP3 ที่ {current_price}"

            elif not pos.get("tp2_hit") and ((is_buy and current_price >= tp2) or (not is_buy and current_price <= tp2)):
                close_qty = round(total_qty * 0.33, 4)
                pnl = (current_price - entry) * close_qty if is_buy else (entry - current_price) * close_qty
                realized += pnl
                self.realized_total_pnl += pnl
                rem_qty = max(0.0, rem_qty - close_qty)
                pos["tp2_hit"] = True
                pos["sl"] = tp1
                state_changed = True
                self.send_telegram(
                    f"🎯 [TP2 HIT] {symbol} แบ่งปิด 33% ที่ {current_price}\n"
                    f"กำไรล็อกแล้ว: {realized:+.2f} USDT | ย้าย SL ล็อกกำไรที่ {tp1}"
                )

            elif not pos.get("tp1_hit") and ((is_buy and current_price >= tp1) or (not is_buy and current_price <= tp1)):
                close_qty = round(total_qty * 0.33, 4)
                pnl = (current_price - entry) * close_qty if is_buy else (entry - current_price) * close_qty
                realized += pnl
                self.realized_total_pnl += pnl
                rem_qty = max(0.0, rem_qty - close_qty)
                pos["tp1_hit"] = True
                pos["sl"] = entry
                state_changed = True
                self.send_telegram(
                    f"✅ [TP1 HIT] {symbol} แบ่งปิด 33% ที่ {current_price}\n"
                    f"กำไรล็อกแล้ว: {realized:+.2f} USDT | ย้าย SL มาบังทุนที่ {entry}"
                )

            pos["remaining_qty"] = round(rem_qty, 4)
            pos["realized_pnl"] = round(realized, 2)

            if closed_all:
                state_changed = True
                pos["closed_at"] = datetime.utcnow().isoformat()
                pos["exit_reason"] = exit_reason
                pos["unrealized_pnl"] = 0.0
                self.closed_history.append(pos)
                self.send_telegram(
                    f"🏁 [CLOSED] ปิดไม้ {symbol} ({exit_reason})\n"
                    f"PnL สุทธิไม้นี้: {realized:+.2f} USDT\n"
                    f"💰 พอร์ตจำลองปัจจุบัน: {self.current_equity:.2f} USDT (สะสม {self.realized_total_pnl:+.2f} USDT)"
                )
            else:
                remaining_positions.append(pos)

        self.paper_positions = remaining_positions
        if state_changed:
            self.storage.save_state(self.paper_positions, self.realized_total_pnl, self.closed_history)


# ---------------------------------------------------------------------------
# 5. ระบบ Async Dynamic Scanner
# ---------------------------------------------------------------------------
DEFAULT_SCAN_SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "DOGE/USDT:USDT",
    "XRP/USDT:USDT", "BNB/USDT:USDT", "PEPE/USDT:USDT", "SUI/USDT:USDT",
    "NEAR/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT", "ADA/USDT:USDT",
    "APT/USDT:USDT", "SHIB/USDT:USDT", "DOT/USDT:USDT"
]

storage = FreePersistentStorage()
pipeline = DeepAlphaQuantPipeline(
    equity_usdt=float(os.getenv("PAPER_EQUITY_USDT", "1000.0")),
    leverage=int(os.getenv("LEVERAGE", "5")),
)
engine = ExecutionEngine(storage=storage)

scanner_stats = {
    "enabled": os.getenv("ENABLE_AUTO_SCANNER", "true").lower() == "true",
    "last_scan_time": None,
    "scan_duration_sec": 0.0,
    "scanned_symbols_count": len(DEFAULT_SCAN_SYMBOLS),
    "current_watchlist": DEFAULT_SCAN_SYMBOLS,
    "total_detected_signals": 0,
    "recent_signals": [],
}


async def get_dynamic_top_symbols(exchange_async, top_n: int = 15) -> list:
    try:
        tickers = await exchange_async.fetch_tickers()
        candidates = []
        for symbol, data in tickers.items():
            if not symbol.endswith("/USDT:USDT"):
                continue
            quote_vol = float(data.get("quoteVolume") or 0.0)
            change_pct = float(data.get("percentage") or 0.0)
            if quote_vol >= 3_000_000 and 1.5 <= change_pct <= 35.0:
                candidates.append((symbol, change_pct, quote_vol))

        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [c[0] for c in candidates[:top_n]]
        return selected if len(selected) >= 5 else DEFAULT_SCAN_SYMBOLS
    except Exception as err:
        logger.warning(f"ใช้รายชื่อเหรียญเริ่มต้นแทน Dynamic Symbols: {err}")
        return DEFAULT_SCAN_SYMBOLS


async def scan_single_symbol(symbol: str, exchange_async, sem: asyncio.Semaphore) -> Optional[dict]:
    async with sem:
        try:
            ohlcv = await exchange_async.fetch_ohlcv(symbol, timeframe="1h", limit=100)
            if not ohlcv or len(ohlcv) < 55:
                return None

            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df = pipeline.compute_market_features(df)
            regime = pipeline.detect_regime(df)

            curr = df.iloc[-1]
            prev = df.iloc[-2]

            vol_ratio = float(curr.get("vol_ratio", 1.0))
            rsi = float(curr.get("rsi_14", 50.0))
            close_price = float(curr["close"])
            ema_20 = float(curr["ema_20"])
            atr = float(curr["atr_14"]) if not pd.isna(curr["atr_14"]) else (close_price * 0.02)

            candle_range = float(curr["high"] - curr["low"])
            upper_wick = float(curr["high"] - curr["close"])
            if candle_range > 0 and (upper_wick / candle_range) > 0.40:
                return None

            if (close_price - ema_20) > (2.2 * atr):
                return None

            if not (52.0 <= rsi <= 73.0):
                return None

            is_bull_breakout = (
                regime == "BULL"
                and vol_ratio >= 1.5
                and (close_price > float(prev["high"]) or vol_ratio >= 2.2)
                and float(prev["close"]) >= float(prev["ema_20"])
            )
            if not is_bull_breakout:
                return None

            funding_rate = 0.0
            try:
                f_data = await exchange_async.fetch_funding_rate(symbol)
                funding_rate = float(f_data.get("fundingRate") or 0.0)
                if funding_rate > 0.0005:
                    return None
            except Exception:
                pass

            plan = pipeline.optimize_tp_sl_with_liquidation(
                entry_price=close_price,
                atr=atr,
                side="buy",
                regime=regime,
                current_equity=engine.current_equity,
            )
            return {
                "symbol": symbol,
                "vol_ratio": round(vol_ratio, 2),
                "rsi": round(rsi, 1),
                "funding_rate": round(funding_rate, 6),
                "regime": regime,
                "plan": plan,
            }
        except Exception:
            return None


# ---------------------------------------------------------------------------
# 6. ระบบ Telegram Bot Command Center (ควบคุมบอทผ่านแชท + ปุ่มกดลัด)
# ---------------------------------------------------------------------------
TELEGRAM_KEYBOARD = {
    "keyboard": [
        [{"text": "📊 เช็คพอร์ต"}, {"text": "📂 ไม้ที่ถืออยู่"}],
        [{"text": "🔥 เหรียญซิ่งวันนี้"}, {"text": "📜 ประวัติเทรด"}],
        [{"text": "⏸️ พักสแกน"}, {"text": "▶️ เปิดสแกน"}, {"text": "🚨 ปิดทุกไม้ทันที"}],
    ],
    "resize_keyboard": True,
}


async def handle_telegram_command(text: str, exchange_async):
    cmd = text.strip()
    cmd_lower = cmd.lower()

    # 1. เช็คสถานะพอร์ต
    if cmd_lower in ("/start", "/help", "help", "เมนู"):
        msg = (
            "🤖 *ศูนย์ควบคุมบอท DeepAlpha*\n\n"
            "กดปุ่มเมนูด้านล่าง หรือพิมพ์คำสั่ง:\n"
            "• `/status` - เช็คยอดเงินและกำไรสุทธิ\n"
            "• `/positions` - ดูเหรียญที่ถืออยู่\n"
            "• `/watchlist` - ดู 15 เหรียญซิ่งที่กำลังสแกน\n"
            "• `/history` - ดูประวัติ 5 ไม้ล่าสุดที่ปิดแล้ว\n"
            "• `/close SOL` - สั่งปิดเหรียญนั้นทันที\n"
            "• `/closeall` - ปิดทุกไม้ในพอร์ตทันที\n"
            "• `/pause` / `/resume` - พัก/เปิดสแกนเนอร์\n"
            "• `/reset` - รีเซ็ตพอร์ตจำลองเริ่มใหม่"
        )
        engine.send_telegram(msg, reply_markup=TELEGRAM_KEYBOARD)

    elif cmd_lower in ("/status", "📊 เช็คพอร์ต"):
        unrealized = sum(float(p.get("unrealized_pnl", 0.0)) for p in engine.paper_positions)
        net_eq = engine.current_equity + unrealized
        scan_state = "🟢 ทำงานปกติ" if scanner_stats["enabled"] else "⏸️ พักการสแกนชั่วคราว"
        msg = (
            f"📊 *สถานะพอร์ตจำลอง (Paper Portfolio)*\n"
            f"• สถานะสแกนเนอร์: {scan_state}\n"
            f"• ทุนเริ่มต้น: {engine.initial_equity:.2f} USDT\n"
            f"• กำไรสะสมที่ปิดแล้ว: {engine.realized_total_pnl:+.2f} USDT\n"
            f"• กำไรไม้ที่ถืออยู่: {unrealized:+.2f} USDT\n"
            f"💰 *มูลค่าพอร์ตสุทธิ: {net_eq:.2f} USDT*\n\n"
            f"• ถือครองอยู่: {len(engine.paper_positions)} ไม้ | ปิดแล้ว: {len(engine.closed_history)} ไม้\n"
            f"• สแกนล่าสุด: {scanner_stats['last_scan_time'] or 'กำลังเริ่ม...'} ({scanner_stats['scan_duration_sec']}s)"
        )
        engine.send_telegram(msg, reply_markup=TELEGRAM_KEYBOARD)

    # 2. ดูไม้ที่ถืออยู่
    elif cmd_lower in ("/positions", "📂 ไม้ที่ถืออยู่"):
        if not engine.paper_positions:
            engine.send_telegram("📂 ปัจจุบันไม่มีไม้ที่เปิดถือครองอยู่ครับ", reply_markup=TELEGRAM_KEYBOARD)
            return
        lines = ["📂 *รายการไม้ที่กำลังถืออยู่:*"]
        for i, p in enumerate(engine.paper_positions, 1):
            sym = p["symbol"].replace(":USDT", "")
            u_pnl = float(p.get("unrealized_pnl", 0.0))
            r_pnl = float(p.get("realized_pnl", 0.0))
            last_p = p.get("last_price", p["entry_price"])
            lines.append(
                f"\n{i}. *{sym}* ({p['side'].upper()})\n"
                f"   เข้า: {p['entry_price']} | ล่าสุด: {last_p}\n"
                f"   เหลือ: {p.get('remaining_qty', p['qty'])} | SL: {p['sl']}\n"
                f"   TP1: {p['tp1_33pct']} {'✅' if p.get('tp1_hit') else '⏳'} | "
                f"TP2: {p['tp2_33pct']} {'✅' if p.get('tp2_hit') else '⏳'} | TP3: {p['tp3_34pct_trail']}\n"
                f"   PnL รอรับรู้: {u_pnl:+.2f} USDT (ล็อกแล้ว: {r_pnl:+.2f})\n"
                f"   👉 พิมพ์ `/close {sym.split('/')[0]}` เพื่อปิดไม้นี้"
            )
        engine.send_telegram("\n".join(lines), reply_markup=TELEGRAM_KEYBOARD)

    # 3. ดูรายชื่อเหรียญซิ่งประจำวัน
    elif cmd_lower in ("/watchlist", "🔥 เหรียญซิ่งวันนี้"):
        coins = [c.replace("/USDT:USDT", "") for c in scanner_stats["current_watchlist"]]
        msg = (
            f"🔥 *15 เหรียญ Dynamic Top Gainers ที่กำลังเฝ้าสแกน:*\n"
            f"{', '.join(coins)}\n\n"
            f"⏱️ ใช้เวลาสแกนรอบละ: {scanner_stats['scan_duration_sec']} วินาที"
        )
        engine.send_telegram(msg, reply_markup=TELEGRAM_KEYBOARD)

    # 4. ดูประวัติไม้ที่ปิดไปแล้ว
    elif cmd_lower in ("/history", "📜 ประวัติเทรด"):
        if not engine.closed_history:
            engine.send_telegram("📜 ยังไม่มีประวัติไม้ที่ปิดจบแล้วครับ", reply_markup=TELEGRAM_KEYBOARD)
            return
        lines = ["📜 *ประวัติ 5 ไม้ล่าสุดที่ปิดแล้ว:*"]
        for p in engine.closed_history[-5:]:
            sym = p["symbol"].replace(":USDT", "")
            pnl = float(p.get("realized_pnl", 0.0))
            icon = "🟢" if pnl >= 0 else "🔴"
            lines.append(f"{icon} *{sym}*: {pnl:+.2f} USDT ({p.get('exit_reason', 'Closed')})")
        engine.send_telegram("\n".join(lines), reply_markup=TELEGRAM_KEYBOARD)

    # 5. พักและเปิดสแกนเนอร์
    elif cmd_lower in ("/pause", "⏸️ พักสแกน"):
        scanner_stats["enabled"] = False
        engine.send_telegram("⏸️ สั่งพักการสแกนหาเหรียญใหม่ชั่วคราวแล้วครับ (ระบบเช็ค TP/SL ของไม้เดิมยังทำงานปกติ)", reply_markup=TELEGRAM_KEYBOARD)

    elif cmd_lower in ("/resume", "▶️ เปิดสแกน"):
        scanner_stats["enabled"] = True
        engine.send_telegram("▶️ เปิดระบบ Auto-Scanner กวาดหาเหรียญซิ่งต่อแล้วครับ!", reply_markup=TELEGRAM_KEYBOARD)

    # 6. สั่งปิดทุกไม้ทันที
    elif cmd_lower in ("/closeall", "🚨 ปิดทุกไม้ทันที"):
        reply = await engine.close_all_positions(exchange_async)
        engine.send_telegram(reply, reply_markup=TELEGRAM_KEYBOARD)

    # 7. สั่งปิดรายเหรียญ เช่น /close SOL
    elif cmd_lower.startswith("/close "):
        coin = cmd.split(" ", 1)[1].strip()
        reply = await engine.manual_close_position(coin, exchange_async)
        engine.send_telegram(reply, reply_markup=TELEGRAM_KEYBOARD)

    # 8. รีเซ็ตพอร์ตจำลองใหม่
    elif cmd_lower == "/reset":
        engine.paper_positions = []
        engine.realized_total_pnl = 0.0
        engine.closed_history = []
        storage.save_state([], 0.0, [])
        engine.send_telegram(
            f"🔄 รีเซ็ตพอร์ตจำลองเรียบร้อย! เริ่มต้นนับหนึ่งใหม่ที่ทุน {engine.initial_equity:.2f} USDT ครับ",
            reply_markup=TELEGRAM_KEYBOARD,
        )


async def telegram_command_listener_loop(exchange_async):
    """ลูปดึงข้อความคำสั่งจาก Telegram แบบ Long-Polling (ไม่กินทรัพยากรเครื่อง)"""
    token = os.getenv("TELEGRAM_TOKEN", "")
    authorized_chat_id = str(os.getenv("TELEGRAM_CHAT_ID", "")).strip()
    if not token or not authorized_chat_id:
        logger.info("ไม่ได้ตั้งค่า TELEGRAM_TOKEN / TELEGRAM_CHAT_ID ข้ามการเปิดรับคำสั่ง Telegram")
        return

    logger.info("📱 เปิดระบบรับคำสั่งควบคุมบอทผ่าน Telegram เรียบร้อย...")
    offset = 0

    # ส่งปุ่มเมนูต้อนรับเมื่อบอทเริ่มทำงาน
    engine.send_telegram(
        "🟢 *บอท DeepAlpha ออนไลน์พร้อมรับคำสั่งแล้วครับ!*\nกดปุ่มเมนูด้านล่างเพื่อเช็คพอร์ตหรือควบคุมบอทได้เลย",
        reply_markup=TELEGRAM_KEYBOARD,
    )

    while True:
        try:
            res = await asyncio.to_thread(
                requests.get,
                f"https://api.telegram.org/bot{token}/getUpdates",
                params={"offset": offset, "timeout": 15},
                timeout=20,
            )
            data = res.json()
            if data.get("ok"):
                for item in data.get("result", []):
                    offset = item["update_id"] + 1
                    msg = item.get("message", {})
                    sender_chat_id = str(msg.get("chat", {}).get("id", ""))
                    text = msg.get("text", "")

                    # ตรวจสอบสิทธิ์: รับคำสั่งเฉพาะจาก Chat ID ของเจ้าของบอทเท่านั้น
                    if sender_chat_id == authorized_chat_id and text:
                        await handle_telegram_command(text, exchange_async)
        except Exception as err:
            logger.debug(f"Telegram polling warning: {err}")
            await asyncio.sleep(5)

        await asyncio.sleep(1)


async def async_master_scanner_loop(exchange_async):
    logger.info("🚀 เริ่มต้นระบบ Async Dynamic Scanner & Paper Exit Engine (ฟรี 100%)...")
    scan_interval = int(os.getenv("SCAN_INTERVAL_SECONDS", "90"))
    max_positions = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
    sem = asyncio.Semaphore(5)

    await asyncio.sleep(3)
    while True:
        loop_start = time.time()
        try:
            await engine.check_paper_exits(exchange_async)

            if scanner_stats["enabled"] and len(engine.paper_positions) < max_positions:
                watchlist = await get_dynamic_top_symbols(exchange_async, top_n=15)
                scanner_stats["current_watchlist"] = watchlist
                scanner_stats["scanned_symbols_count"] = len(watchlist)

                open_symbols = {p["symbol"] for p in engine.paper_positions}
                targets = [s for s in watchlist if s not in open_symbols]

                tasks = [scan_single_symbol(sym, exchange_async, sem) for sym in targets]
                results = await asyncio.gather(*tasks)

                current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                scanner_stats["last_scan_time"] = current_time
                scanner_stats["scan_duration_sec"] = round(time.time() - loop_start, 2)

                for res in results:
                    if res and len(engine.paper_positions) < max_positions:
                        sym = res["symbol"]
                        engine.execute_trade_plan(sym, res["plan"])
                        sig_info = {
                            "time": current_time,
                            "symbol": sym,
                            "vol_ratio": res["vol_ratio"],
                            "rsi": res["rsi"],
                            "funding_rate": res["funding_rate"],
                            "regime": res["regime"],
                            "entry_price": res["plan"]["entry_price"],
                        }
                        scanner_stats["recent_signals"] = ([sig_info] + scanner_stats["recent_signals"])[:10]
                        scanner_stats["total_detected_signals"] += 1
            else:
                scanner_stats["last_scan_time"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                scanner_stats["scan_duration_sec"] = round(time.time() - loop_start, 2)

        except Exception as err:
            logger.error(f"Error in master scanner cycle: {err}")

        await asyncio.sleep(scan_interval)


# ---------------------------------------------------------------------------
# 7. FastAPI Server & Endpoints
# ---------------------------------------------------------------------------
app = FastAPI(title="Custom DeepAlpha Async Quant Engine")


@app.on_event("startup")
async def startup_event():
    ex_class = getattr(ccxt_async, engine.exchange_name)
    exchange_async = ex_class({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    asyncio.create_task(async_master_scanner_loop(exchange_async))
    asyncio.create_task(telegram_command_listener_loop(exchange_async))


@app.get("/")
@app.get("/status")
def get_status():
    unrealized_total = sum(float(p.get("unrealized_pnl", 0.0)) for p in engine.paper_positions)
    return {
        "status": "online",
        "paper_mode": engine.paper_mode,
        "exchange": engine.exchange_name,
        "portfolio": {
            "initial_equity_usdt": engine.initial_equity,
            "realized_pnl_usdt": round(engine.realized_total_pnl, 2),
            "unrealized_pnl_usdt": round(unrealized_total, 2),
            "net_equity_usdt": round(engine.current_equity + unrealized_total, 2),
            "open_positions_count": len(engine.paper_positions),
            "closed_trades_count": len(engine.closed_history),
        },
        "auto_scanner": {
            "enabled": scanner_stats["enabled"],
            "last_scan_time": scanner_stats["last_scan_time"],
            "scan_duration_sec": scanner_stats["scan_duration_sec"],
            "scanned_symbols_count": scanner_stats["scanned_symbols_count"],
            "current_watchlist": scanner_stats["current_watchlist"],
            "total_detected_signals": scanner_stats["total_detected_signals"],
            "recent_signals": scanner_stats["recent_signals"][:5],
        },
        "open_positions": engine.paper_positions,
        "closed_history": engine.closed_history[-10:],
    }


@app.post("/webhook/tradingview")
async def tradingview_webhook(request: Request):
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

    try:
        ohlcv = engine.exchange.fetch_ohlcv(symbol, timeframe="1h", limit=100)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    except Exception as err:
        raise HTTPException(status_code=502, detail=f"Exchange fetch error: {err}")

    df = pipeline.compute_market_features(df)
    regime = pipeline.detect_regime(df)
    last_close = float(df["close"].iloc[-1])
    last_atr = float(df["atr_14"].iloc[-1]) if not pd.isna(df["atr_14"].iloc[-1]) else (last_close * 0.02)

    plan = pipeline.optimize_tp_sl_with_liquidation(
        entry_price=last_close, atr=last_atr, side=action, regime=regime, current_equity=engine.current_equity
    )
    return engine.execute_trade_plan(symbol, plan)


if __name__ == "__main__":
    port = int(os.getenv("PORT") or os.getenv("WEBHOOK_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
