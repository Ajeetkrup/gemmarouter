"""Self-contained launcher module for the Live Head-to-Head Decision Playground."""

from __future__ import annotations

import webbrowser

from . import _playground_server


def launch_playground(
    host: str = "127.0.0.1",
    port: int = 7860,
    our_repo: str = "Ajeet-001/embeddinggemma2-decision-router",
    open_browser: bool = True,
    preload: bool = True,
) -> None:
    """Start the interactive 3-column Live Head-to-Head Decision Playground server."""
    server = _playground_server.create_server(host=host, port=port, our_repo=our_repo, preload=preload)
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}"
    print("=" * 78)
    print("GEMMAROUTER LIVE HEAD-TO-HEAD DECISION PLAYGROUND READY")
    print(f"URL: {url}")
    print("=" * 78)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
