import os,time,requests,pandas as pd,numpy as np
BASE=os.getenv("MARKET_DATA_BASE","https://fapi.binance.com").rstrip("/")
SYMBOLS=("BTCUSDT","ETHUSDT")
def fetch(symbol,interval,limit=500,end_time=None):
    if symbol not in SYMBOLS or interval not in ("15m","1h"): raise ValueError("Unsupported market")
    params={"symbol":symbol,"interval":interval,"limit":min(limit,1500)}
    if end_time is not None: params["endTime"]=int(end_time)
    r=requests.get(BASE+"/fapi/v1/klines",params=params,timeout=20);r.raise_for_status()
    rows=r.json()
    if not isinstance(rows,list): raise ValueError("Invalid API response")
    cols=["open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_base","taker_quote","ignore"]
    d=pd.DataFrame(rows,columns=cols)
    for c in ("open","high","low","close","volume","open_time","close_time"): d[c]=pd.to_numeric(d[c],errors="raise")
    return d.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
def closed(d,now=None):
    now=int(time.time()*1000) if now is None else now
    return d[d.close_time<now].copy().reset_index(drop=True)
def indicators(d):
    d=d.copy()
    for n in (20,50,200): d["ema"+str(n)]=d.close.ewm(span=n,adjust=False).mean()
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
    return dict(id=f"{symbol}:{int(v.open_time)}:{side}",symbol=symbol,side=side,entry=entry,stop=stop,target=target,qty=qty,notional=notional,risk=risk,rsi=float(v.rsi),volume_ratio=float(v.volume/v.volavg),close_time=int(v.close_time))
