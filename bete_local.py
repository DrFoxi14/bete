#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tiny localhost UI for bete_engine.py. Uses only the Python standard library."""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from bete_engine import process_image

ROOT = Path(__file__).resolve().parent
WEB_FILE = ROOT / "index.html"
DATA_ROOT = ROOT / ".bete_local" / "runs"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEBUG_FILES = {
    "debug_1_cerneala.png",
    "debug_2_schelet.png",
    "debug_3_rezultat.png",
}
RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class BeteHandler(BaseHTTPRequestHandler):
    server_version = "BeteLocal/1.0"

    def _send_bytes(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, data, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            if not WEB_FILE.is_file():
                self._send_bytes(500, b"index.html is missing", "text/plain; charset=utf-8")
                return
            self._send_bytes(200, WEB_FILE.read_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/health":
            self._send_json(200, {"ok": True, "engine": "bete_engine.py"})
            return
        if parsed.path.startswith("/api/files/"):
            self._serve_file(parsed.path)
            return
        self._send_json(404, {"error": "Ruta inexistenta."})

    def _serve_file(self, url_path: str) -> None:
        parts = [unquote(part) for part in url_path.split("/") if part]
        if len(parts) != 4 or parts[0:2] != ["api", "files"]:
            self._send_json(404, {"error": "Fișier inexistent."})
            return
        run_id, filename = parts[2], parts[3]
        if not RUN_ID_RE.fullmatch(run_id) or Path(filename).name != filename:
            self._send_json(404, {"error": "Adresă invalidă."})
            return
        allowed_input = filename in {"input.png", "input.jpg", "input.jpeg"}
        allowed_json = filename == "input_rezultate.json"
        if filename not in DEBUG_FILES and not allowed_input and not allowed_json:
            self._send_json(403, {"error": "Fișierul nu este expus de server."})
            return
        file_path = DATA_ROOT / run_id / filename
        if not file_path.is_file():
            self._send_json(404, {"error": "Fișierul nu a fost găsit."})
            return
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        self._send_bytes(200, file_path.read_bytes(), content_type)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/analyze":
            self._send_json(404, {"error": "Rută API inexistentă."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Content-Length invalid."})
            return
        if length <= 0:
            self._send_json(400, {"error": "Nu ai trimis nicio fotografie."})
            return
        if length > MAX_UPLOAD_BYTES:
            self._send_json(413, {"error": "Fotografia depășește limita de 25 MB."})
            return

        query = parse_qs(parsed.query)
        original_name = Path(query.get("filename", ["fotografie.jpg"])[0]).name
        extension = Path(original_name).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            self._send_json(415, {"error": "Format acceptat: PNG, JPG sau JPEG."})
            return

        px_per_cm = None
        raw_scale = query.get("px_per_cm", [""])[0].strip()
        if raw_scale:
            try:
                px_per_cm = float(raw_scale)
            except ValueError:
                self._send_json(400, {"error": "Calibrarea px/cm trebuie să fie numerică."})
                return
            if px_per_cm <= 0:
                self._send_json(400, {"error": "Calibrarea px/cm trebuie să fie pozitivă."})
                return

        image_bytes = self.rfile.read(length)
        if len(image_bytes) != length:
            self._send_json(400, {"error": "Încărcarea fotografiei a fost incompletă."})
            return

        run_id = uuid.uuid4().hex
        run_dir = DATA_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        input_path = run_dir / ("input" + extension)
        input_path.write_bytes(image_bytes)
        try:
            result = process_image(input_path, run_dir, px_per_cm)
        except Exception as exc:
            self._send_json(422, {"error": "Analiza a eșuat: " + str(exc)[:500]})
            return

        result["debug_images"] = {
            "debug_1_cerneala": "/api/files/" + run_id + "/debug_1_cerneala.png",
            "debug_2_schelet": "/api/files/" + run_id + "/debug_2_schelet.png",
            "debug_3_rezultat": "/api/files/" + run_id + "/debug_3_rezultat.png",
        }
        result["original_image_url"] = "/api/files/" + run_id + "/input" + extension
        result["json_download_url"] = "/api/files/" + run_id + "/input_rezultate.json"
        result["run_id"] = run_id
        self._send_json(200, result)

    def log_message(self, fmt: str, *args) -> None:
        sys.stdout.write("[bete-local] " + self.address_string() + " - " + (fmt % args) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Interfață web locală pentru motorul Bete")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Implicit 127.0.0.1; nu schimba decât dacă vrei acces din rețea.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true",
                        help="Nu deschide automat browserul.")
    args = parser.parse_args()
    if not WEB_FILE.is_file():
        print("Eroare: index.html nu este lângă bete_local.py.", file=sys.stderr)
        return 2
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        server = ThreadingHTTPServer((args.host, args.port), BeteHandler)
    except OSError as exc:
        print("Nu pot porni serverul pe " + args.host + ":" + str(args.port) + ": " + str(exc),
              file=sys.stderr)
        return 2
    url = "http://" + args.host + ":" + str(server.server_address[1])
    print("\nBete Engine este pornit.")
    print("Deschide: " + url)
    print("Fotografiile și rezultatele rămân local în .bete_local/runs/")
    print("Oprește serverul cu Ctrl+C.\n")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nOprire server...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
