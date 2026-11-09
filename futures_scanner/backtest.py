import argparse,json,pandas as pd
from .core import fetch,closed,signal,PROVIDER
from .paper import settle
def history(symbol,interval,bars):
    chunks=[];end=None;count=0
    while count<bars:
        n=min(499,bars-count);d=fetch(symbol,interval,n,end)
        if d.empty:break
        chunks.append(d);count+=len(d);end=int(d.open_time.iloc[0])-1
        if len(d)<n:break
    if not chunks:raise RuntimeError("No historical candles")
    return pd.concat(chunks).sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
def run(symbol,bars=2000,capital=50.0):
    a=closed(history(symbol,"15m",bars));h=closed(history(symbol,"1h",max(500,bars//4+220)))
    equity=capital;peak=capital;dd=0;trades=[];pos=None;losses={}
    for i in range(220,len(a)-1):
        candle=a.iloc[i].to_dict();day=int(candle["close_time"]//86400000)
        if pos:
            result=settle(pos,candle)
            if result:
                equity+=result["pnl"];trades.append(result)
                if result["pnl"]<0:losses[day]=losses.get(day,0)+1
                peak=max(peak,equity);dd=max(dd,(peak-equity)/peak if peak else 0);pos=None
        if pos or equity<=0 or losses.get(day,0)>=3:continue
        hh=h[h.close_time<=a.close_time.iloc[i]]
        s=signal(a.iloc[:i+1],hh,symbol,risk=min(.5,equity*.01))
        if not s:continue
        price=float(a.open.iloc[i+1])
        if (s["side"]=="LONG" and price<=s["stop"]) or (s["side"]=="SHORT" and price>=s["stop"]):continue
        qty=min(.5,equity*.01)/(abs(price-s["stop"])+(price+s["stop"])*.0007)
        if qty*price>min(equity*1.8,90):continue
        pos={**s,"entry":price,"qty":qty}
    gains=sum(max(0,t["pnl"]) for t in trades);loss=-sum(min(0,t["pnl"]) for t in trades)
    result={"provider":PROVIDER,"symbol":symbol,"trades":len(trades),"net_pnl":sum(t["pnl"] for t in trades),"end_equity":equity,"profit_factor":gains/loss if loss else None,"max_drawdown_pct":dd*100,"note":"Simulation only; Hyperliquid perpetual data (USDC), funding and minimum lots not modeled"}
    print(json.dumps(result,indent=2));return result
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--symbol",choices=["BTCUSDT","ETHUSDT"],default="BTCUSDT");p.add_argument("--bars",type=int,default=2000)
    x=p.parse_args();run(x.symbol,x.bars)
