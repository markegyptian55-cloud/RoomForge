from __future__ import annotations


class Room3DError(Exception):
    """Base error for worker domain failures."""


class ManifestValidationError(Room3DError):
    """A manifest is missing required fields or violates its contract."""


class GeometryError(Room3DError):
    """A geometric input cannot produce a valid transform or projection."""


class StageExecutionError(Room3DError):
    """An isolated stage failed to start, finish, or return successfully."""


class QualityGateError(Room3DError):
    """A quality gate or progress contract cannot be satisfied."""
