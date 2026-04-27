from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import subprocess
import shutil
import tempfile

app = FastAPI()

# Path to the polyglot PDF in the dist directory
RESUME_PDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dist", "resume.pdf")

def _run_pdf(pdf_path: str, *args):
    """Make the PDF executable and run it with optional args."""
    # Copy to a temp location so we can chmod without touching the original
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copy2(pdf_path, tmp.name)
        tmp_path = tmp.name

    try:
        os.chmod(tmp_path, 0o755)
        return subprocess.run(
            [tmp_path, *args],
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.post("/api/chat")
@app.post("/")
async def chat_with_resume(message: str = Form(...)):
    if not os.path.isfile(RESUME_PDF):
        raise HTTPException(status_code=500, detail="Resume PDF not found on server.")

    try:
        result = _run_pdf(RESUME_PDF, message)
        output = (result.stdout or result.stderr).strip()
        if not output:
            output = "(no response)"
        return JSONResponse({"response": output})

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="PDF engine timed out.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
