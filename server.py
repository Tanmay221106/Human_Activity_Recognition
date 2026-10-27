import http.server
import socketserver
import json
import os
import mimetypes
import urllib.parse
from har_engine import HAREngine

PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

engine = HAREngine()

class HARRequestHandler(http.server.BaseHTTPRequestHandler):
    def end_headers(self):
        # Allow cross-origin and disable caching for development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == '/' or path == '/index.html':
            self.serve_file('index.html', 'text/html')
            return
        elif path == '/style.css':
            self.serve_file('style.css', 'text/css')
            return
        elif path == '/app.js':
            self.serve_file('app.js', 'application/javascript')
            return
        elif path == '/api/model-info':
            self.send_json(engine.metadata)
            return
        elif path == '/api/sample-files':
            samples = [
                {"name": "walking.csv", "label": "Walking (Full Session)", "size": "834 KB"},
                {"name": "standing.csv", "label": "Standing (Full Session)", "size": "677 KB"},
                {"name": "sitting.csv", "label": "Sitting (Full Session)", "size": "656 KB"},
                {"name": "sleeping.csv", "label": "Sleeping (Full Session)", "size": "361 KB"},
                {"name": "cleaned_sensor_data.csv", "label": "Cleaned Walking Excerpt", "size": "14 KB"},
            ]
            self.send_json({"files": samples})
            return
        else:
            # Check if file exists in workspace
            local_path = os.path.join(BASE_DIR, path.lstrip('/'))
            if os.path.isfile(local_path):
                mime, _ = mimetypes.guess_type(local_path)
                self.serve_file(path.lstrip('/'), mime or 'application/octet-stream')
                return

        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)

        if path == '/api/analyze-file':
            try:
                body = json.loads(post_data.decode('utf-8'))
                filename = body.get('filename')
                if not filename:
                    self.send_json({"success": False, "error": "No filename specified"}, 400)
                    return

                filepath = os.path.join(BASE_DIR, os.path.basename(filename))
                if not os.path.exists(filepath):
                    self.send_json({"success": False, "error": f"File '{filename}' not found"}, 404)
                    return

                # Read raw bytes (limit to ~100k bytes for instant responsive analysis if very large)
                with open(filepath, 'rb') as f:
                    raw_bytes = f.read(120000)

                result = engine.analyze(raw_bytes)
                result['active_file'] = filename
                self.send_json(result)
            except Exception as e:
                self.send_json({"success": False, "error": str(e)}, 500)
            return

        elif path == '/api/analyze-upload':
            try:
                # Direct upload analysis
                result = engine.analyze(post_data)
                result['active_file'] = "Uploaded File"
                self.send_json(result)
            except Exception as e:
                self.send_json({"success": False, "error": str(e)}, 500)
            return

        self.send_error(404, "Endpoint not found")

    def serve_file(self, filename, content_type):
        filepath = os.path.join(BASE_DIR, filename)
        if not os.path.exists(filepath):
            self.send_error(404, f"File {filename} not found")
            return
        with open(filepath, 'rb') as f:
            content = f.read()
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, data, code=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

def run():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), HARRequestHandler) as httpd:
        print(f"\n=======================================================")
        print(f"  HAR Sentinel Server running on http://127.0.0.1:{PORT}")
        print(f"  MLF (Random Forest) | INS (SHA-256) | DAV (Waveform)")
        print(f"=======================================================\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")

if __name__ == '__main__':
    run()
