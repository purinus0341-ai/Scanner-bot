import requests
from unittest.mock import patch
from futures_scanner.core import fetch,closed
class Response:
    status_code=200
    def raise_for_status(self):pass
    def json(self):return {"retCode":0,"result":{"list":[["1700000900000","101","103","100","102","5","500"],["1700000000000","100","102","99","101","7","700"]]}}
def test_bybit_candle_conversion():
    with patch("futures_scanner.core.requests.get",return_value=Response()) as req:
        d=fetch("BTCUSDT","15m",2)
    assert list(d.open_time)==[1700000000000,1700000900000]
    assert d.close_time.iloc[0]==1700000899999
    assert d.close.iloc[-1]==102
    assert req.call_args.kwargs["params"]["interval"]=="15"
def test_closed_candles_only():
    with patch("futures_scanner.core.requests.get",return_value=Response()):
        d=fetch("BTCUSDT","15m",2)
    assert len(closed(d,now=1700000900000))==1
def test_restricted_api_fails_closed():
    r=requests.Response();r.status_code=451;r.url="https://api.bybit.com/v5/market/kline"
    with patch("futures_scanner.core.requests.get",return_value=r):
        try:fetch("BTCUSDT","15m")
        except RuntimeError as e:assert "no signals sent" in str(e)
        else:assert False
