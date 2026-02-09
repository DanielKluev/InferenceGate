"""
Outflow component - forwards requests to the real AI model endpoint.

Key classes: `OutflowClient`
"""

from inference_gate.outflow.client import OutflowClient

__all__ = ["OutflowClient"]
