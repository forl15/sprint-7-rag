import json
import time
import urllib.request

OLLAMA_URL = "http://ollama:11434"

for _ in range(180):
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        names = {m.get("name") for m in data.get("models", [])}
        if "qwen3:8b" in names:
            break
    except Exception:
        pass
    time.sleep(2)

req = urllib.request.Request(
    f"{OLLAMA_URL}/api/pull",
    data=json.dumps({"model": "qwen3:8b"}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        _ = resp.read()
except Exception:
    pass

import uvicorn
from chat_bot.chat_bot import app

uvicorn.run(app, host="0.0.0.0", port=8000)
