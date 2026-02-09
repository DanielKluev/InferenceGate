"""
Operating modes for the InferenceGate proxy.

Key classes: `Mode`
"""

from enum import Enum


class Mode(str, Enum):
    """
    Operating modes for the InferenceGate proxy.

    RECORD_AND_REPLAY: Record new inferences and replay existing ones (default mode).
        When a matching inference is found in storage, it is replayed.
        When no match is found, the request is forwarded to the real AI endpoint,
        the response is captured and stored for future replays.
    REPLAY_ONLY: Only replay inferences from local storage.
        If a matching inference is not found, returns an error response.
    """

    RECORD_AND_REPLAY = "record-and-replay"
    REPLAY_ONLY = "replay-only"
