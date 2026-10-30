r"""Launch the LeanMoE dashboard on this computer only.

    .\.venv\Scripts\python.exe -m src.dashboard            (then open http://127.0.0.1:8770)
    .\.venv\Scripts\python.exe -m src.dashboard --open     (also opens the browser)
    .\.venv\Scripts\python.exe -m src.dashboard --port 8771

It listens on 127.0.0.1 only; there is no option to expose it on a network.
Stop it with Ctrl+C. It never downloads model files.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import src.engine  # noqa: E402,F401  (first: sets the OpenBLAS thread default before numpy loads)

import argparse  # noqa: E402
import webbrowser  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=None, help="port on 127.0.0.1 (default 8770)")
    ap.add_argument("--open", action="store_true", help="open the dashboard in the default browser")
    args = ap.parse_args()

    from src.dashboard.server import DEFAULT_PORT, HOST, serve

    port = DEFAULT_PORT if args.port is None else args.port
    try:
        server, engine = serve(port)
    except OSError as e:
        print(f"Could not listen on {HOST}:{port} ({e}). Another program may be using it; try --port 8771.")
        return 1
    url = f"http://{HOST}:{server.server_address[1]}/"
    status = engine.status()
    print(f"LeanMoE dashboard: {url}  (this computer only; Ctrl+C to stop)")
    print(f"OpenBLAS threads {status['threads']['openblas']}, numba threads {status['threads']['numba']}")
    if status["real_store"]["available"]:
        print(f"Real model store: {status['real_store']['path']}")
    else:
        print("Real model store: not available (the synthetic engine demonstration still works).")
        print(f"  {status['real_store']['problem']}")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping ...")
        if engine.active is not None:
            engine.active.cancel.cancel()
    finally:
        server.server_close()
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
