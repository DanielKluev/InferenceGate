"""
Outflow component - forwards requests to the real AI model endpoint.

Key classes: ``OutflowClient``, ``OutflowRouter``, ``UpstreamConfig``
"""

from inference_gate.outflow.client import OutflowClient
from inference_gate.outflow.model_router import OutflowRouter, UpstreamConfig

__all__ = ["OutflowClient", "OutflowRouter", "UpstreamConfig"]
