"""
HAR Sentinel UI Launcher
Launches the local HTTP server and automatically opens the user's browser.
"""

import webbrowser
import threading
import time
import server

def open_browser():
    time.sleep(0.8)
    url = f"http://127.0.0.1:{server.PORT}/"
    print(f"Opening browser at {url} ...")
    webbrowser.open(url)

if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    server.run()
