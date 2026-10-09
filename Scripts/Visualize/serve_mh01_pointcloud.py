#!/usr/bin/env python3
"""Serve the MH01 viewer over localhost and open it in the default browser."""
from __future__ import annotations

import argparse
import http.server
import socketserver
import webbrowser
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    url = f"http://127.0.0.1:{args.port}/Results/MH01_pointcloud_3d.html"
    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(*a, directory=str(root), **kw)
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Serving MH01 viewer at {url} (Ctrl+C to stop)")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")


if __name__ == "__main__":
    main()
