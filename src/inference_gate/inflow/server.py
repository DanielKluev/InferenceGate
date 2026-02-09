"""
Inflow HTTP server that accepts incoming requests from clients.

Built using `aiohttp` and designed to be fully asynchronous.
The server parses incoming requests and passes them to the Router
for further processing.

Key classes: `InflowServer`
"""

import json
import logging
from typing import Any

from aiohttp import web

from inference_gate.recording.storage import CachedResponse
from inference_gate.router.router import Router


class InflowServer:
    """
    HTTP server that accepts incoming OpenAI-compatible API requests.

    Listens on the configured host/port and forwards all requests
    to the `Router` for routing decisions (replay or forward upstream).
    """

    def __init__(self, host: str, port: int, router: Router) -> None:
        """
        Initialize the inflow server.

        `host` is the address to bind to (e.g. "127.0.0.1").
        `port` is the port number to listen on.
        `router` is the Router instance that handles request routing.
        """
        self.log = logging.getLogger("InflowServer")
        self.host = host
        self.port = port
        self.router = router
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    def _create_app(self) -> web.Application:
        """
        Create the aiohttp web application with route handlers.

        Returns the configured `web.Application`.
        """
        app = web.Application()
        app.router.add_route("GET", "/health", self._handle_health)
        # Catch-all route for proxying API requests
        app.router.add_route("*", "/{path:.*}", self._handle_proxy)
        return app

    async def start(self) -> None:
        """
        Start the HTTP server and begin listening for requests.
        """
        self._app = self._create_app()
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        self.log.info("InflowServer started on http://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        """
        Stop the HTTP server and clean up resources.
        """
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        self._site = None
        self._app = None
        self.log.info("InflowServer stopped")

    async def _handle_health(self, request: web.Request) -> web.Response:
        """
        Handle health check endpoint.

        Returns JSON with status and current mode.
        """
        data = {"status": "healthy", "mode": self.router.mode.value}
        return web.json_response(data)

    async def _handle_proxy(self, request: web.Request) -> web.Response:
        """
        Handle all proxied API requests.

        Parses the incoming request, delegates to the Router,
        and builds the appropriate HTTP response.
        """
        method = request.method
        path = f"/{request.match_info['path']}"

        # Parse request headers
        headers: dict[str, str] = dict(request.headers)

        # Parse request body if present
        body: dict[str, Any] | None = None
        if method in ("POST", "PUT", "PATCH"):
            try:
                body = await request.json()
            except (json.JSONDecodeError, ValueError):
                body = None

        # Parse query parameters
        query_params: dict[str, str] | None = None
        if request.query_string:
            query_params = dict(request.query)

        # Route the request
        self.log.debug("Received %s %s", method, path)
        cached_response = await self.router.route_request(method=method, path=path, headers=headers, body=body, query_params=query_params)

        # Build and return the response
        return self._build_response(cached_response)

    def _build_response(self, cached_response: CachedResponse) -> web.Response:
        """
        Build an aiohttp Response from a CachedResponse.

        Handles both streaming (SSE) and standard JSON responses.
        """
        if cached_response.is_streaming and cached_response.chunks:
            # Reconstruct the streaming response as a single body
            # For replay, we send all chunks concatenated
            body_bytes = "".join(cached_response.chunks).encode("utf-8")
            return web.Response(body=body_bytes, status=cached_response.status_code, content_type="text/event-stream")

        # Standard JSON response
        if cached_response.body is not None:
            return web.json_response(data=cached_response.body, status=cached_response.status_code)

        return web.Response(status=cached_response.status_code)
