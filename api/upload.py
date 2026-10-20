"""POST /api/chat (form field `message`) -> {"response": ...} from the 83M PDFLLM persona model.

The model (pdfllm/) reads the resume in context and answers in first person; it runs on ONNX Runtime
on CPU, with the attention/SSM cache, so one answer takes about 10 s on Vercel.
"""

import sys
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdfllm.engine import answer  # noqa: E402

app = FastAPI()


@app.post("/api/chat")
@app.post("/")
def chat_with_resume(message: str = Form(...)):  # sync: FastAPI runs it in a thread, off the event loop
    try:
        out = answer(message[:500], cached=True, max_new=64)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"response": out["answer"] or "(no response)"})
