"""
InferenceGate - AI inference replay for testing, debugging and development.

Provides tools to capture, store, and replay AI model inferences, enabling
developers to simulate scenarios and validate model behavior without repeated
live inferences.
"""

from inference_gate.inference_gate import InferenceGate
from inference_gate.modes import Mode
from inference_gate.recording.storage import CacheStorage

__version__ = "0.1.0"
__all__ = ["CacheStorage", "InferenceGate", "Mode", "__version__"]
