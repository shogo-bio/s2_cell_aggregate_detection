"""Typed errors. Every failure mode that a user can cause has its own class so
callers can distinguish a bad config from a corrupt artifact from a model problem.
"""

from __future__ import annotations


class S2AdhesionError(Exception):
    """Base for every error raised by this package."""


class ConfigError(S2AdhesionError):
    """Configuration is malformed, incomplete, or internally inconsistent."""


class ContractViolation(S2AdhesionError):
    """An in-memory structure violates the canonical axis/dtype/unit conventions."""


class ArtifactError(S2AdhesionError):
    """An on-disk artifact is missing, incomplete, or fails validation."""


class ArtifactBindingError(ArtifactError):
    """A label artifact does not match the image artifact it claims to describe.

    Raised on any mismatch of shape, spacing, field identity, or content hash.
    This is the guard that stops labels from one preprocessing run being silently
    measured against a different image.
    """


class SegmentationError(S2AdhesionError):
    """Segmentation could not be performed."""


class MLDependencyError(SegmentationError):
    """A required ML package is absent or its major version does not match config.

    Never raised by the measurement path -- measurement must work with no ML
    packages installed at all.
    """


class MeasurementError(S2AdhesionError):
    """A measurement could not be computed and no meaningful null exists."""
