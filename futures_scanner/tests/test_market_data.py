import requests
from unittest.mock import patch
from futures_scanner.core import fetch,closed
class Response:
    status_code=200
    def raise_for_status(self):pass
    def json(self):
        return [
            {"t":1700000000000,"T":1700000899999,"o":"100","h":"102","l":"99","c":"101","v":"7"},
            {"t":1700000900000,"T":1700001799999,"o":"101","h":"103","l":"100","c":"102","v":"5"}
        ]
def test_hyperliquid_candle_conversion():
    with patch("futures_scanner.core.requests.post",return_value=Response()) as req:
        d=fetch("BTCUSDT","15m",2,end_time=1700001800000)
    assert list(d.open_time)==[1700000000000,1700000900000]
    assert d.close_time.iloc[0]==1700000899999
    assert d.close.iloc[-1]==102
    assert req.call_args.kwargs["json"]["req"]["coin"]=="BTC"
    assert req.call_args.kwargs["json"]["req"]["interval"]=="15m"
def test_closed_candles_only():
    with patch("futures_scanner.core.requests.post",return_value=Response()):
        d=fetch("BTCUSDT","15m",2,end_time=1700001800000)
    assert len(closed(d,now=1700000900000))==1
def test_restricted_api_fails_closed():
    r=requests.Response();r.status_code=403;r.url="https://api.hyperliquid.xyz/info"
    with patch("futures_scanner.core.requests.post",return_value=r):
        try:fetch("BTCUSDT","15m")
        except RuntimeError as e:assert "no signals sent" in str(e)
        else:assert False
