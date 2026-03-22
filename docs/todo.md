# To-Do List

## 1) Core Functionality

- [x] Pytest integration - built-in pytest plugin with `inference_gate_url` fixture, auto-server launch, replay/record modes, and cassette sanitization. See [docs/pytest-integration.md](pytest-integration.md).

## 2) Development Helpers

- [ ] Add option to imitate faults at the given rate, up to 1.0 (100% of requests)
- [ ] Add option to add random latency to responses, with configurable mean and variance