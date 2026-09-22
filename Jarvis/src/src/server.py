import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import subprocess

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"JARVIS is alive!")

def run_http_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), SimpleHandler)
    print(f"HTTP server running on port {port}")
    server.serve_forever()

def run_agent():
    # Aapka main agent command jo LiveKit se connect karega
    subprocess.run(["uv", "run", "src/agent.py", "start"])

if __name__ == "__main__":
    # HTTP server ko background thread mein chalayein taaki Render ka health check pass ho jaye
    t = threading.Thread(target=run_http_server, daemon=True)
    t.start()
    
    # Main thread mein agent run karein
    run_agent() 