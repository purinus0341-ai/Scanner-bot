import json,os
from pathlib import Path
from .core import SYMBOLS,fetch,closed,signal
from .notify import send,format_signal
STATE=Path(os.getenv("SCANNER_STATE_PATH","futures_scanner/state.json"))
def run():
    STATE.parent.mkdir(parents=True,exist_ok=True)
    state=json.loads(STATE.read_text()) if STATE.exists() else {"sent":[]}
    state.setdefault("sent",[])
    for symbol in SYMBOLS:
        a=closed(fetch(symbol,"15m"));h=closed(fetch(symbol,"1h"))
        s=signal(a,h,symbol)
        if s and s["id"] not in state["sent"]:
            if send(format_signal(s)):state["sent"].append(s["id"])
    state["sent"]=state["sent"][-1000:]
    STATE.write_text(json.dumps(state,indent=2)+"\n")
    print("Scan complete; alerts recorded:",len(state["sent"]))
if __name__=="__main__":run()
