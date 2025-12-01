"""Operating modes for the inference replay proxy."""

from enum import Enum


class Mode(str, Enum):
    """Operating modes for the inference replay proxy.

    RECORD: Always proxy to upstream API and cache responses
    DEVELOPMENT: Replay from cache if available, otherwise record
    REPLAY: Only replay from cache, fail if not found
    """

    RECORD = "record"
    DEVELOPMENT = "development"
    REPLAY = "replay"
