"""Chat with the 83M PDFLLM persona model (form field `message`):
  POST /api/chat         -> {"response": ...}
  POST /api/chat/stream  -> NDJSON events from pdfllm.engine.stream: each answer token with the PDF words
                            its copy head read (the site's heatmap)

The model (pdfllm/) reads the resume in context and answers in first person; it runs on ONNX Runtime
on CPU, with the attention/SSM cache, so one answer takes about 10 s on Vercel.
"""

import json
import sys
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdfllm.engine import answer, stream  # noqa: E402

app = FastAPI()


@app.post("/api/chat")
@app.post("/")
def chat_with_resume(message: str = Form(...)):  # sync: FastAPI runs it in a thread, off the event loop
    try:
        out = answer(message[:500], cached=True, max_new=64)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"response": out["answer"] or "(no response)"})


@app.post("/api/chat/stream")
def chat_stream(message: str = Form(...)):
    lines = (json.dumps(ev) + "\n" for ev in stream(message[:500], max_new=64))
    return StreamingResponse(lines, media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
