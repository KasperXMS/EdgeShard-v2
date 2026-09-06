"""Profile persistence (Phase 2 spec §43-§46).

``ProfileStore`` is the repository interface the controller and CLI talk
to; SQLite v1 lives behind it and SQL never leaks into domain or
controller code.
"""
