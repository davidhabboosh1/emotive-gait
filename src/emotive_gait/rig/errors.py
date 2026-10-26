"""Errors for the rig pipeline."""

from __future__ import annotations


class RigError(ValueError):
    """A user-input problem with a rig; the message names the offending item."""
