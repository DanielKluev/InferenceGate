"""
Core class that glues all other components together and serves as the entrypoint.

Manages the lifecycle of the application: creates, starts, and stops all
components (Inflow, Router, Outflow, Recording).

Key classes: `InferenceGate`
"""

import asyncio
import logging

from inference_gate.inflow.server import InflowServer
from inference_gate.modes import Mode
from inference_gate.outflow.client import OutflowClient
from inference_gate.recording.storage import CacheStorage
from inference_gate.router.router import Router


class InferenceGate:
    """
    Central orchestrator for the InferenceGate proxy system.

    Creates and manages all components: InflowServer, Router, OutflowClient,
    and CacheStorage. Handles startup and shutdown lifecycle.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8080, mode: Mode = Mode.RECORD_AND_REPLAY,
                 cache_dir: str = ".inference_cache", upstream_base_url: str = "https://api.openai.com", api_key: str | None = None,
                 web_ui: bool = False, web_ui_port: int = 8081) -> None:
        """
        Initialize InferenceGate with configuration.

        `host` and `port` configure the local proxy server address.
        `mode` determines behavior: record-and-replay or replay-only.
        `cache_dir` is the directory for storing cached inferences.
        `upstream_base_url` is the real AI API endpoint URL.
        `api_key` is the API key for upstream authentication.
        `web_ui` enables the optional web dashboard.
        `web_ui_port` is the port for the web UI server.
        """
        self.log = logging.getLogger("InferenceGate")
        self.host = host
        self.port = port
        self.mode = mode
        self.cache_dir = cache_dir
        self.upstream_base_url = upstream_base_url
        self.api_key = api_key
        self.web_ui = web_ui
        self.web_ui_port = web_ui_port

        # Components (created during start)
        self._storage: CacheStorage | None = None
        self._outflow: OutflowClient | None = None
        self._router: Router | None = None
        self._server: InflowServer | None = None
        self._webui_server: "WebUIServer | None" = None

    def _create_components(self) -> None:
        """
        Create all system components based on configuration.

        Creates CacheStorage, OutflowClient (if needed), Router, InflowServer, and optionally WebUIServer.
        """
        self._storage = CacheStorage(self.cache_dir)

        # OutflowClient is only needed for RECORD_AND_REPLAY mode
        if self.mode == Mode.RECORD_AND_REPLAY:
            self._outflow = OutflowClient(upstream_base_url=self.upstream_base_url, api_key=self.api_key)
        else:
            self._outflow = None

        self._router = Router(mode=self.mode, storage=self._storage, outflow=self._outflow)
        self._server = InflowServer(host=self.host, port=self.port, router=self._router)

        # WebUIServer is optional
        if self.web_ui:
            from inference_gate.webui.server import WebUIServer
            self._webui_server = WebUIServer(host=self.host, port=self.web_ui_port, storage=self._storage, mode=self.mode,
                                            cache_dir=self.cache_dir, upstream_base_url=self.upstream_base_url, proxy_host=self.host,
                                            proxy_port=self.port)

    async def start(self) -> None:
        """
        Start InferenceGate and all its components.

        Creates components, starts the outflow client (if applicable),
        and starts the inflow HTTP server and WebUI server (if enabled).
        """
        self.log.info("Starting InferenceGate in %s mode", self.mode.value)
        self._create_components()

        # Start outflow client if we need upstream access
        if self._outflow is not None:
            await self._outflow.start()

        # Start the inflow HTTP server
        assert self._server is not None
        await self._server.start()

        # Start the WebUI server if enabled
        if self._webui_server is not None:
            await self._webui_server.start()

        self.log.info("InferenceGate ready on http://%s:%d", self.host, self.port)
        if self._webui_server is not None:
            self.log.info("WebUI dashboard available at http://%s:%d", self.host, self.web_ui_port)

    async def stop(self) -> None:
        """
        Stop InferenceGate and all its components.

        Stops the inflow server, WebUI server (if enabled), and outflow client in order.
        """
        self.log.info("Stopping InferenceGate")

        if self._webui_server is not None:
            await self._webui_server.stop()

        if self._server is not None:
            await self._server.stop()

        if self._outflow is not None:
            await self._outflow.stop()

        self.log.info("InferenceGate stopped")

    async def run_forever(self) -> None:
        """
        Start InferenceGate and run until interrupted.

        Blocks until a KeyboardInterrupt or cancellation occurs,
        then performs clean shutdown.
        """
        await self.start()
        try:
            # Run until interrupted
            while True:
                await asyncio.sleep(3600)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await self.stop()

    @property
    def storage(self) -> CacheStorage | None:
        """
        Get the CacheStorage instance, if created.
        """
        return self._storage