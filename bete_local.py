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

from bete_engine import process_image as process_image_page1
from bete_engine_page2 import process_image as process_image_page2
from bete_engine_page3 import process_image as process_image_page3
from geometry_layer2 import process_layer1_json
from geometry_layer3a_v51 import process_layer3a

ROOT = Path(__file__).resolve().parent
WEB_FILE = ROOT / "index.html"
DATA_ROOT = ROOT / ".bete_local" / "runs"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEBUG_FILES = {
    "debug_1_cerneala.png",
    "debug_2_schelet.png",
    "debug_3_rezultat.png",
    "debug_3_toate_betele.png",
    "debug_4_context.png",
    "debug_1_binar_exact.png",
    "debug_2_dss_toate.png",
    "debug_3_verificare_xor.png",
    "debug_1_dss_exact.png",
    "debug_2_tipuri.png",
    "debug_3_alungire.png",
    "debug_4_verificare_100pct.png",
    "debug_l2_1_grile_si_bare.png",
    "debug_l2_2_poduri_si_ancore.png",
    "debug_l2_3_coridoare_si_siruri.png",
    "debug_l3a_1_grid_metric_support.png",
    "debug_l3a_2_contacts_geometry.png",
    "debug_l3a_3_accounting_zones.png",
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
            self._send_json(200, {"ok": True, "engines": {"1": "bete_engine.py", "2": "bete_engine_page2.py", "3": "Straturile 0 + 1 + 2 + Geometry Layer 3A v5.1"}})
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
        allowed_json = filename in {"input_rezultate.json", "input_bete_v3.json", "input_dss_lossless.json", "input_geometry_layer1.json", "input_geometry_layer2.json", "input_geometry_layer3a.json"}
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

        page = query.get("page", ["1"])[0]
        if page not in {"1", "2", "3"}:
            self._send_json(400, {"error": "Pagina trebuie să fie 1, 2 sau 3."})
            return

        run_id = uuid.uuid4().hex
        run_dir = DATA_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        input_path = run_dir / ("input" + extension)
        input_path.write_bytes(image_bytes)
        try:
            if page == "1":
                result = process_image_page1(input_path, run_dir, px_per_cm)
                debug_names = [
                    "debug_1_cerneala.png",
                    "debug_2_schelet.png",
                    "debug_3_rezultat.png",
                ]
                json_name = "input_rezultate.json"
                engine_name = "bete_engine.py"
            elif page == "2":
                result = process_image_page2(input_path, run_dir, px_per_cm)
                debug_names = [
                    "debug_1_cerneala.png",
                    "debug_2_schelet.png",
                    "debug_3_toate_betele.png",
                    "debug_4_context.png",
                ]
                json_name = "input_bete_v3.json"
                engine_name = "bete_engine_page2.py"
            else:
                # Page 3: stratul 0 (DSS exact) + stratul 1 (partiție geometrică).
                result = process_image_page3(input_path, run_dir, scale=1.0)
                debug_names = [
                    "debug_1_dss_exact.png",
                    "debug_2_tipuri.png",
                    "debug_3_alungire.png",
                    "debug_4_verificare_100pct.png",
                ]
                json_name = "input_geometry_layer1.json"
                engine_name = "bete_engine_page3.py + geometry_layer2.py + geometry_layer3a_v51.py"
                # API-ul motorului întoarce sumarul; încărcăm primitivele/joncțiunile
                # din JSON-ul complet generat în același director al execuției.
                full_result_path = Path(result["json_file"]).resolve()
                if full_result_path.parent != run_dir.resolve() or full_result_path.name != "input_geometry_layer1.json":
                    raise RuntimeError("Stratul 1 a produs un JSON în afara execuției curente.")
                full_result = json.loads(full_result_path.read_text(encoding="utf-8"))
                result["primitives"] = full_result.get("primitives", [])
                result["junctions"] = full_result.get("junctions", [])

                # Stratul 2 este strict condiționat de trecerea Straturilor 0 + 1.
                proof = full_result.get("lossless_verification", {})
                if int(proof.get("layer0_dss_xor_error_pixels", -1)) != 0:
                    raise RuntimeError("Stratul 2 blocat: verificarea XOR a Stratului 0 nu este zero.")
                if int(proof.get("layer1_uncovered_ink_pixels", -1)) != 0 or proof.get("guaranteed_100pct_coverage") is not True:
                    raise RuntimeError("Stratul 2 blocat: Stratul 1 nu a confirmat acoperirea 100%.")

                # Funcția primește numai JSON-ul validat din Stratul 1; imaginea este
                # folosită de ea exclusiv ca fundal pentru imaginile de debug.
                layer2_result = process_layer1_json(full_result_path, run_dir, image_override=str(input_path))
                audit = layer2_result.get("accounting_verification", {})
                if (audit.get("guaranteed_100pct_accounting") is not True
                        or int(audit.get("unaccounted_primitives", -1)) != 0
                        or int(audit.get("duplicate_primitives", -1)) != 0):
                    raise RuntimeError("Stratul 2 nu a confirmat contabilitatea completă a primitivelor.")
                result["layer2"] = {
                    key: value for key, value in layer2_result.items()
                    if key not in {"debug_images", "json_file"}
                }
                result["layer2_debug_images"] = {
                    Path(name).stem: "/api/files/" + run_id + "/" + Path(name).name
                    for name in layer2_result["debug_images"].values()
                }
                layer2_json_path = Path(layer2_result.get("json_file", "")).resolve()
                if layer2_json_path.parent != run_dir.resolve() or layer2_json_path.name != "input_geometry_layer2.json":
                    raise RuntimeError("Stratul 2 a produs un JSON în afara execuției curente.")

                # Stratul 3A v5.1 primește exclusiv JSON-urile validate ale Straturilor 1 și 2.
                layer3a_result = process_layer3a(
                    full_result_path, layer2_json_path, run_dir, image_override=str(input_path)
                )
                layer3a_json_path = Path(layer3a_result.get("json_file", "")).resolve()
                if layer3a_json_path.parent != run_dir.resolve() or layer3a_json_path.name != "input_geometry_layer3a.json":
                    raise RuntimeError("Stratul 3A a produs un JSON în afara execuției curente.")
                audit3a = layer3a_result.get("audit_verification", {})
                if (audit3a.get("guaranteed_exact_once_accounting") is not True
                        or int(audit3a.get("missing_primitives", -1)) != 0
                        or int(audit3a.get("duplicate_primitive_assignments", -1)) != 0
                        or int(audit3a.get("missing_molecules", -1)) != 0
                        or int(audit3a.get("duplicate_molecule_assignments", -1)) != 0):
                    raise RuntimeError("Stratul 3A nu a confirmat auditul strict count == 1.")

                result["layer3a"] = {
                    "schema_version": layer3a_result.get("schema_version"),
                    "upstream_gates": layer3a_result.get("upstream_gates", {}),
                    "audit_verification": audit3a,
                    "totals": layer3a_result.get("totals", {}),
                    "grids": [{
                        "grid_id": grid.get("grid_id"),
                        "metrics": grid.get("metrics", {}),
                        "counts": grid.get("counts", {}),
                    } for grid in layer3a_result.get("grids", [])],
                }
                result["layer3a_debug_images"] = {
                    Path(name).stem: "/api/files/" + run_id + "/" + Path(name).name
                    for name in layer3a_result.get("debug_images", {}).values()
                }
                result["layer1_json_download_url"] = "/api/files/" + run_id + "/input_geometry_layer1.json"
                result["layer2_json_download_url"] = "/api/files/" + run_id + "/input_geometry_layer2.json"
                result["layer3a_json_download_url"] = "/api/files/" + run_id + "/input_geometry_layer3a.json"
                json_name = "input_geometry_layer3a.json"
        except Exception as exc:
            self._send_json(422, {"error": "Analiza a eșuat: " + str(exc)[:500]})
            return

        result["debug_images"] = {
            Path(name).stem: "/api/files/" + run_id + "/" + name
            for name in debug_names
        }
        result["original_image_url"] = "/api/files/" + run_id + "/input" + extension
        result["json_download_url"] = "/api/files/" + run_id + "/" + json_name
        result["run_id"] = run_id
        result["page"] = page
        result["engine"] = engine_name
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
