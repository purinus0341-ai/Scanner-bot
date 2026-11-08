from futures_scanner.paper import settle
def test_stop_wins_when_both_hit():
    p={"side":"LONG","entry":100,"stop":99,"target":102,"qty":1}
    assert settle(p,{"low":98,"high":103})["reason"]=="STOP"
def test_no_exit():
    p={"side":"LONG","entry":100,"stop":99,"target":102,"qty":1}
    assert settle(p,{"low":99.5,"high":101}) is None
