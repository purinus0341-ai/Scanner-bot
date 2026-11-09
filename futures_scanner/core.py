import os,time,requests,pandas as pd,numpy as np
BASE=os.getenv("MARKET_DATA_BASE","https://api.hyperliquid.xyz").rstrip("/")
SYMBOLS=("BTCUSDT","ETHUSDT")
PROVIDER="Hyperliquid USDC-margined perpetual (public market data)"
def fetch(symbol,interval,limit=500,end_time=None):
    if symbol not in SYMBOLS or interval not in ("15m","1h"):raise ValueError("Unsupported market")
    if not 1<=int(limit)<=500:raise ValueError("Limit must be 1..500")
    duration=900000 if interval=="15m" else 3600000
    end=int(time.time()*1000) if end_time is None else int(end_time)
    start=end-int(limit)*duration
    coin=symbol.replace("USDT","")
    payload={"type":"candleSnapshot","req":{"coin":coin,"interval":interval,"startTime":start,"endTime":end}}
    try:
        r=requests.post(BASE+"/info",json=payload,timeout=25)
        r.raise_for_status()
    except requests.HTTPError as e:
        if r.status_code in (403,451):raise RuntimeError(f"{PROVIDER} market data unavailable (HTTP {r.status_code}); no signals sent") from e
        raise
    rows=r.json()
    if not isinstance(rows,list):raise RuntimeError("Invalid Hyperliquid candles response")
    if not rows:
        return pd.DataFrame(columns=["open_time","open","high","low","close","volume","close_time"])
    d=pd.DataFrame([{"open_time":v["t"],"close_time":v["T"],"open":v["o"],"high":v["h"],"low":v["l"],"close":v["c"],"volume":v["v"]} for v in rows])
    for c in ("open_time","close_time","open","high","low","close","volume"):d[c]=pd.to_numeric(d[c],errors="raise")
    d=d[(d.open_time>=start)&(d.open_time<end)].sort_values("open_time").drop_duplicates("open_time").tail(int(limit)).reset_index(drop=True)
    if d.empty:raise RuntimeError("No candles in requested time window")
    if (d.open_time.diff().dropna()<=0).any():raise RuntimeError("Invalid candle chronology")
    return d
def closed(d,now=None):
    now=int(time.time()*1000) if now is None else now
    return d[d.close_time<now].copy().reset_index(drop=True)
def indicators(d):
    d=d.copy()
    for n in (20,50,200):d["ema"+str(n)]=d.close.ewm(span=n,adjust=False).mean()
    delta=d.close.diff();gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    d["rsi"]=100-100/(1+gain/loss.replace(0,np.nan))
    prev=d.close.shift()
    tr=pd.concat([d.high-d.low,(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1)
    d["atr"]=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    d["volavg"]=d.volume.shift().rolling(20).mean()
    return d
def signal(d15,d1h,symbol,risk=0.5):
    if len(d15)<220 or len(d1h)<220:return None
    a=indicators(d15);b=indicators(d1h);v=a.iloc[-1];p=a.iloc[-2]
    hb=b[b.close_time<=v.close_time]
    if hb.empty or not np.isfinite(v.rsi) or not np.isfinite(v.atr) or v.atr<=0:return None
    h=hb.iloc[-1]
    long=h.close>h.ema200 and v.ema20>v.ema50 and p.low<=p.ema20*1.002 and 50<=v.rsi<=65 and v.close>p.high and v.volume>1.2*v.volavg
    short=h.close<h.ema200 and v.ema20<v.ema50 and p.high>=p.ema20*.998 and 35<=v.rsi<=50 and v.close<p.low and v.volume>1.2*v.volavg
    if not (long or short):return None
    entry=float(v.close);side="LONG" if long else "SHORT";recent=a.iloc[-6:]
    stop=min(float(recent.low.min()),entry-float(v.atr)) if long else max(float(recent.high.max()),entry+float(v.atr))
    dist=abs(entry-stop);target=entry+(2*dist if long else -2*dist)
    if dist<=0 or target<=0 or dist/entry>.04:return None
    qty=risk/(dist+(entry+stop)*.0007);notional=qty*entry
    if notional>90:return None
    return dict(id=f"hyperliquid:{symbol}:{int(v.open_time)}:{side}",symbol=symbol,side=side,entry=entry,stop=stop,target=target,qty=qty,notional=notional,risk=risk,rsi=float(v.rsi),volume_ratio=float(v.volume/v.volavg),close_time=int(v.close_time))
