"""
WebUI HTTP server that serves the dashboard and API endpoints.

Built using `aiohttp` and designed to be fully asynchronous.
Serves both the static React SPA and JSON API endpoints.

Key classes: `WebUIServer`
"""

import logging
from pathlib import Path

from aiohttp import web

from inference_gate.modes import Mode
from inference_gate.recording.storage import CacheStorage
from inference_gate.webui.api import WebUIAPI


class WebUIServer:
    """
    HTTP server for the WebUI Dashboard.

    Serves the React SPA static files and provides JSON API endpoints
    for accessing cache data, statistics, and configuration.
    """

    def __init__(self, host: str, port: int, storage: CacheStorage, mode: Mode, cache_dir: str, upstream_base_url: str | None,
                 proxy_host: str, proxy_port: int) -> None:
        """
        Initialize the WebUI server.

        `host` is the address to bind to (e.g. "127.0.0.1").
        `port` is the port number to listen on.
        `storage` is the CacheStorage instance for accessing cached data.
        `mode` is the current operation mode.
        `cache_dir` is the cache directory path.
        `upstream_base_url` is the upstream API URL (None in REPLAY_ONLY mode).
        `proxy_host` and `proxy_port` are the main proxy server configuration.
        """
        self.log = logging.getLogger("WebUIServer")
        self.host = host
        self.port = port
        self.storage = storage
        self.mode = mode
        self.cache_dir = cache_dir
        self.upstream_base_url = upstream_base_url
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port

        # API handler
        self.api = WebUIAPI(storage=storage, mode=mode, cache_dir=cache_dir, upstream_base_url=upstream_base_url, host=proxy_host,
                            port=proxy_port)

        # Runtime state
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    def _get_static_dir(self) -> Path:
        """
        Get the path to the static files directory.

        Returns the absolute path to the static directory.
        """
        # The static directory is in the same package as this module
        return Path(__file__).parent / "static"

    def _create_app(self) -> web.Application:
        """
        Create the aiohttp web application with route handlers.

        Returns the configured `web.Application`.
        """
        app = web.Application()

        # API routes
        app.router.add_route("GET", "/api/cache", self.api.get_cache_list)
        app.router.add_route("GET", "/api/cache/{entry_id}", self.api.get_cache_entry)
        app.router.add_route("GET", "/api/stats", self.api.get_stats)
        app.router.add_route("GET", "/api/config", self.api.get_config)

        # Static file serving
        static_dir = self._get_static_dir()
        assets_dir = static_dir / "assets"
        index_path = static_dir / "index.html"

        if static_dir.exists() and assets_dir.is_dir() and index_path.is_file():
            # Serve static assets (without following symlinks to avoid exposing files outside the static directory)
            app.router.add_static("/assets", assets_dir, name="static_assets", show_index=False)

            # Serve root-level public files (vite.svg, etc.)
            for public_file in static_dir.glob("*.svg"):
                file_name = public_file.name
                app.router.add_route("GET", f"/{file_name}", self._create_static_file_handler(public_file))

            # Serve index.html for all non-API routes (SPA fallback)
            app.router.add_route("GET", "/", self._handle_index)
            app.router.add_route("GET", "/{path:.*}", self._handle_spa_fallback)
        else:
            self.log.warning(
                "Static frontend not fully available (static_dir=%s, assets_dir=%s, index_html=%s)",
                static_dir,
                assets_dir,
                index_path,
            )
            # Add a fallback route that returns a message
            app.router.add_route("GET", "/{path:.*}", self._handle_no_static)

        return app

    def _create_static_file_handler(self, file_path: Path):
        """
        Create a handler function for serving a static file.

        Returns a coroutine that serves the file when called.
        """

        async def handler(request: web.Request) -> web.Response:
            return web.FileResponse(file_path)

        return handler

    async def _handle_index(self, request: web.Request) -> web.Response:
        """
        Handle GET / - serve index.html.

        Returns the React SPA's main HTML file.
        """
        static_dir = self._get_static_dir()
        index_path = static_dir / "index.html"

        if index_path.exists():
            return web.FileResponse(index_path)
        else:
            return await self._handle_no_static(request)

    async def _handle_spa_fallback(self, request: web.Request) -> web.Response:
        """
        Handle all non-API routes - serve index.html for SPA routing.

        This allows React Router to handle client-side routing.
        """
        path = request.match_info.get("path", "")

        # Don't fallback for API routes or asset routes
        if path.startswith("api/") or path.startswith("assets/"):
            raise web.HTTPNotFound()

        # Serve index.html for SPA routing
        return await self._handle_index(request)

    async def _handle_no_static(self, request: web.Request) -> web.Response:
        """
        Handle requests when static files are not available.

        Returns a helpful message explaining that the frontend needs to be built.
        """
        message = """
        <html>
        <head><title>InferenceGate WebUI</title></head>
        <body>
            <h1>InferenceGate WebUI</h1>
            <p>The WebUI frontend has not been built yet.</p>
            <p>To build the frontend, run:</p>
            <pre>
cd webui-frontend
npm install
npm run build
            </pre>
            <p>API endpoints are still available at <code>/api/*</code></p>
        </body>
        </html>
        """
        return web.Response(text=message, content_type="text/html")

    async def start(self) -> None:
        """
        Start the HTTP server and begin listening for requests.
        """
        self._app = self._create_app()
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        self.log.info("WebUIServer started on http://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        """
        Stop the HTTP server and clean up resources.
        """
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        self._site = None
        self._app = None
        self.log.info("WebUIServer stopped")
