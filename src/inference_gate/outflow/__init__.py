"""
Outflow component - forwards requests to the real AI model endpoint.

Key classes: ``OutflowClient``, ``OutflowRouter``, ``EndpointConfig``, ``ModelRoute``
"""

from inference_gate.outflow.client import OutflowClient
from inference_gate.outflow.model_router import EndpointConfig, ModelRoute, OutflowRouter

__all__ = ["OutflowClient", "OutflowRouter", "EndpointConfig", "ModelRoute"]
