def settle(position,candle,fee=.0005,slippage=.0002):
    side=position["side"];entry=position["entry"];stop=position["stop"];target=position["target"];qty=position["qty"]
    hit_stop=candle["low"]<=stop if side=="LONG" else candle["high"]>=stop
    hit_target=candle["high"]>=target if side=="LONG" else candle["low"]<=target
    if not (hit_stop or hit_target):return None
    price=stop if hit_stop else target
    gross=(price-entry)*qty*(1 if side=="LONG" else -1)
    return {"exit":price,"reason":"STOP" if hit_stop else "TARGET","pnl":gross-(entry+price)*qty*(fee+slippage)}
