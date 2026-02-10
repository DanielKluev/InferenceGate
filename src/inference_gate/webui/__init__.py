"""
Web UI Dashboard for InferenceGate.

Provides an optional web-based user interface for browsing cached inference entries,
viewing statistics, and inspecting request/response details.

Key classes: `WebUIServer`
"""

from inference_gate.webui.server import WebUIServer

__all__ = ["WebUIServer"]
