#!/usr/bin/env python3
"""Local dev server — serves static files + the FastAPI /api/chat endpoint.
Avoids vercel dev's IPC buffer limit on the 17 MB resume.pdf.
"""

import os, sys, mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs
import subprocess, shutil, tempfile, json

PORT = 3000
ROOT = os.path.dirname(os.path.abspath(__file__))
RESUME_PDF = os.path.join(ROOT, "dist", "resume.pdf")


def run_pdf(message: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copy2(RESUME_PDF, tmp.name)
        tmp_path = tmp.name
    try:
        os.chmod(tmp_path, 0o755)
        result = subprocess.run(
            [tmp_path, message], capture_output=True, text=True, timeout=30
        )
        output = (result.stdout or result.stderr).strip()
        return output if output else "(no response)"
    except subprocess.TimeoutExpired:
        return "(timed out)"
    except Exception as e:
        return f"(error: {e})"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


class DevHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def do_POST(self):
        if self.path.startswith("/api/chat"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode()

            # Parse multipart form data (simple extraction)
            message = ""
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" in content_type:
                boundary = content_type.split("boundary=")[1].strip()
                parts = body.split("--" + boundary)
                for part in parts:
                    if 'name="message"' in part:
                        # Value is after the double newline
                        message = part.split("\r\n\r\n", 1)[1].rsplit("\r\n", 1)[0]
                        break
            elif "application/x-www-form-urlencoded" in content_type:
                params = parse_qs(body)
                message = params.get("message", [""])[0]

            if not message:
                self._json(400, {"detail": "No message provided"})
                return

            response_text = run_pdf(message)
            self._json(200, {"response": response_text})
        else:
            self._json(404, {"detail": "Not found"})

    def do_GET(self):
        # Serve index.html for root and unknown paths (SPA fallback)
        path = self.path.split("?")[0]
        file_path = os.path.join(ROOT, path.lstrip("/"))

        if path == "/":
            file_path = os.path.join(ROOT, "index.html")
        elif not os.path.isfile(file_path):
            file_path = os.path.join(ROOT, "index.html")

        if os.path.isfile(file_path):
            mime, _ = mimetypes.guess_type(file_path)
            self.send_response(200)
            self.send_header("Content-Type", mime or "application/octet-stream")
            self.send_header("Content-Length", os.path.getsize(file_path))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self._json(404, {"detail": "Not found"})

    def _json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"  {args[0]}")


if __name__ == "__main__":
    print(f"\n  ✦ dev server running at http://localhost:{PORT}\n")
    HTTPServer(("", PORT), DevHandler).serve_forever()
