"""`python -m app`: the two-array show at http://localhost:8000. Static files only: the page (`app/web/`) and,
under /show/, the bundle `app.export` wrote (APP_SHOW_DIR). `app/compose.yaml` serves the same two folders with nginx.
"""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from utils.config import env
from utils.config import path as env_path

WEB = Path(__file__).parent / "web"


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript"}   # Windows registry says text/plain

    def translate_path(self, path):
        local = Path(super().translate_path(path))                # already cleaned of "..", inside WEB
        rel = local.relative_to(WEB.resolve())
        return str(env_path("APP_SHOW_DIR", "data/app/show") / rel.relative_to("show") if rel.parts[:1] == ("show",) else local)

    def log_message(self, *a):
        pass


def serve(port=None):
    return ThreadingHTTPServer(("127.0.0.1", env("APP_PORT", 8000) if port is None else port), partial(Handler, directory=str(WEB)))


def main():
    srv = serve()
    print(f"http://localhost:{srv.server_address[1]}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
