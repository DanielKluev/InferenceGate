# Command-Line Interface (CLI) Reference

## Recording Modes

### Record New, Replay Existing (Default)

In this mode, `InferenceGate` will attempt to replay inferences from the local storage. If a matching inference is not found, it will forward the request to the real AI model endpoint, capture the response, and store it in the storage for future replays.

### Replay Only

In this mode, `InferenceGate` will only attempt to replay inferences from the local storage. If a matching inference is not found, it will return an error response indicating that the inference could not be replayed.